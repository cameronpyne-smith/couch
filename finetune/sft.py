import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from finetune.lora import apply_lora, save_lora_adapter
from optimiser import AdamW, clip_gradients
from train import get_learning_rate, log_losses, make_directory

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
IGNORE = -100
LOSS_CHUNK = 512
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj")


def load_examples(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def encode(example, tokeniser):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["prompt"]},
    ]
    prompt_ids = tokeniser.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False, enable_thinking=False)
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


def load_model(rank, alpha, device="cuda", name=MODEL, bits=16, layers=None):
    model = load_base(name, bits, device)
    trainable = apply_lora(model, rank, alpha, PROJECTIONS, layers)
    return model, trainable

def load_base(name, bits, device="cuda"):
    if bits == 16:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device)
    else:
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=config, dtype=torch.bfloat16, device_map=device)

    return model


def to_device(batch, pad_id, device):
    return tuple(tensor.to(device) for tensor in collate(batch, pad_id))


def chunk_loss(head, states, targets):
    return F.cross_entropy(head(states).float(), targets, reduction="sum")


def token_loss(head, states, targets, chunk=LOSS_CHUNK):
    starts = range(0, len(targets), chunk)
    losses = (checkpoint(chunk_loss, head, states[start : start + chunk], targets[start : start + chunk], use_reentrant=False) for start in starts)
    return sum(losses, torch.zeros((), device=states.device))


def compute_loss(model, input_ids, attention_mask, labels):
    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    targets = labels[:, 1:]
    keep = targets != IGNORE
    return token_loss(model.lm_head, hidden[:, :-1][keep], targets[keep])


def labelled_tokens(batches):
    return sum(label != IGNORE for batch in batches for _, labels in batch for label in labels)


def evaluate(model, batches, pad_id, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in batches:
            input_ids, attention_mask, labels = to_device(batch, pad_id, device)
            total += compute_loss(model, input_ids, attention_mask, labels).item()
    model.train()
    return total / labelled_tokens(batches)


def train(args):
    device = "cuda"
    tokeniser = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokeniser.pad_token_id
    train_encoded = [encode(example, tokeniser) for example in load_examples(args.data / "train.jsonl")]
    val_batches = make_batches([encode(example, tokeniser) for example in load_examples(args.data / "val.jsonl")], args.max_tokens, 0)
    batches_per_epoch = len(make_batches(train_encoded, args.max_tokens, 0))
    max_steps = args.epochs * -(-batches_per_epoch // args.accumulation)

    layers = range(*args.layers) if args.layers else None
    model, trainable = load_model(args.rank, args.alpha, device, args.model, args.bits, layers)
    if args.checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    optimiser = AdamW(trainable, weight_decay=0.0)
    directory = make_directory(args.output_dir)
    print("run", directory, "optimiser steps", max_steps, "trainable", sum(p.numel() for p in trainable) / 1e6, "M")

    stats = []
    best_val = float("inf")
    step = 0
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
        batches = make_batches(train_encoded, args.max_tokens, epoch + 1)
        for start in range(0, len(batches), args.accumulation):
            group = batches[start : start + args.accumulation]
            group_tokens = labelled_tokens(group)
            for batch in group:
                input_ids, attention_mask, labels = to_device(batch, pad_id, device)
                loss = compute_loss(model, input_ids, attention_mask, labels)
                (loss / group_tokens).backward()
                train_sum += loss.item()
                tokens += int(attention_mask.sum())
            train_count += group_tokens
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
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--bits", type=int, default=16, choices=(16, 4))
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--layers", type=int, nargs=2, metavar=("FIRST", "STOP"), help="train LoRA only on blocks FIRST up to but not including STOP; all blocks without it")

    train(parser.parse_args())
