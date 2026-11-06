import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune.lora import apply_lora, save_lora_adapter
from optimiser import AdamW, clip_gradients
from train import get_learning_rate, log_losses, make_directory

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA = Path.home() / "training-data" / "code" / "pr-reviews-verifier"
RUNS = DATA / "runs"
IGNORE = -100
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_examples(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def encode(example, tokeniser):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["prompt"]},
    ]
    prompt_ids = tokeniser.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)
    answer_ids = tokeniser(example["answer"], add_special_tokens=False).input_ids + [tokeniser.eos_token_id]
    return prompt_ids + answer_ids, [IGNORE] * len(prompt_ids) + answer_ids


def make_batches(encoded, max_tokens, seed):
    batches = [[]]
    for pair in sorted(encoded, key=lambda pair: len(pair[0])):
        if batches[-1] and (len(batches[-1]) + 1) * len(pair[0]) > max_tokens:
            batches.append([])
        batches[-1].append(pair)
    random.Random(seed).shuffle(batches)
    return batches


def collate(batch, pad_id):
    width = max(len(ids) for ids, _ in batch)
    input_ids = [ids + [pad_id] * (width - len(ids)) for ids, _ in batch]
    attention_mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch]
    labels = [row + [IGNORE] * (width - len(row)) for _, row in batch]
    return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)


def load_model(rank, alpha, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    trainable = apply_lora(model, rank, alpha, PROJECTIONS)
    return model, trainable


def to_device(batch, pad_id, device):
    return tuple(tensor.to(device) for tensor in collate(batch, pad_id))


def evaluate(model, batches, pad_id, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in batches:
            input_ids, attention_mask, labels = to_device(batch, pad_id, device)
            total += model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss.item()
    model.train()
    return total / len(batches)


def train(args):
    device = "cuda"
    tokeniser = AutoTokenizer.from_pretrained(MODEL)
    pad_id = tokeniser.pad_token_id
    train_encoded = [encode(example, tokeniser) for example in load_examples(args.data / "train.jsonl")]
    val_batches = make_batches([encode(example, tokeniser) for example in load_examples(args.data / "val.jsonl")], args.max_tokens, 0)
    batches_per_epoch = len(make_batches(train_encoded, args.max_tokens, 0))
    max_steps = args.epochs * batches_per_epoch // args.accumulation

    model, trainable = load_model(args.rank, args.alpha, device)
    optimiser = AdamW(trainable, weight_decay=0.0)
    directory = make_directory(args.data / "runs")
    print("run", directory, "optimiser steps", max_steps, "trainable", sum(p.numel() for p in trainable) / 1e6, "M")

    stats = []
    best_val = float("inf")
    step = micro = 0
    tokens = 0
    train_sum = train_count = 0
    started = time.time()

    def log(lr):
        nonlocal best_val, train_sum, train_count
        val = evaluate(model, val_batches, pad_id, device)
        train_mean = train_sum / train_count if train_count else float("nan")
        train_sum = train_count = 0
        if val < best_val:
            best_val = val
            save_lora_adapter(model, directory)
        stats.append({"step": step, "train": train_mean, "val": val, "lr": lr})
        log_losses(stats, directory)
        print(f"step {step} train {train_mean:.4f} val {val:.4f} lr {lr:.2e}")

    log(0.0)
    for epoch in range(args.epochs):
        for batch in make_batches(train_encoded, args.max_tokens, epoch + 1):
            input_ids, attention_mask, labels = to_device(batch, pad_id, device)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            (loss / args.accumulation).backward()
            train_sum += loss.item()
            train_count += 1
            tokens += int(attention_mask.sum())
            micro += 1
            if micro % args.accumulation:
                continue
            lr = get_learning_rate(step, warm_up_steps=args.warmup, max_steps=max_steps, max_learning_rate=args.lr, min_learning_rate=args.lr / 10)
            clip_gradients(trainable, args.clip)
            optimiser.learning_rate = lr
            optimiser.step()
            optimiser.zero_grad()
            step += 1
            if step % args.eval_interval == 0:
                log(lr)
    if step % args.eval_interval:
        log(lr)

    elapsed = time.time() - started
    print(f"best val {best_val:.4f}")
    print(f"{elapsed:.0f} s, {tokens / elapsed:.0f} tokens/s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=24)
    parser.add_argument("--data", type=Path, default=DATA)
    train(parser.parse_args())
