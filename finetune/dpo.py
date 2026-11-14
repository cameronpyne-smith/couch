import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from finetune.lora import load_lora_adapter, save_lora_adapter
from finetune.sft import IGNORE, MODEL, encode, load_examples, load_model, to_device
from optimiser import AdamW, clip_gradients
from train import get_learning_rate, log_losses, make_directory


def sequence_log_probs(model, input_ids, attention_mask, labels):
    hidden = model.model(
        input_ids=input_ids, attention_mask=attention_mask
    ).last_hidden_state
    targets = labels[:, 1:]
    keep = targets != IGNORE
    logits = model.lm_head(hidden[:, :-1][keep]).float()
    losses = F.cross_entropy(logits, targets[keep], reduction="none")
    grid = torch.zeros_like(targets, dtype=losses.dtype)
    grid[keep] = losses
    summed_loss = grid.sum(dim=1)
    return -summed_loss


# policy = model being trained, reference = frozen starting model
def dpo_loss(
    policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta
):
    chosen_gain = policy_chosen - reference_chosen
    rejected_gain = policy_rejected - reference_rejected
    margin = chosen_gain - rejected_gain
    x = beta * margin
    loss = -F.logsigmoid(x)
    return loss.mean()


def pairs_from_samples(samples, correct):
    right = []
    wrong = []
    for sample in samples:
        if sample["verdict"] == "none":
            continue

        if sample["verdict"] == correct:
            right.append(sample["text"])
        else:
            wrong.append(sample["text"])

    return list(zip(right, wrong))


def encode_pair(row, tokeniser):
    chosen = encode(row | {"answer": row["chosen"]}, tokeniser)
    rejected = encode(row | {"answer": row["rejected"]}, tokeniser)
    return chosen, rejected


def pair_log_probs(model, pair, pad_id, device):
    input_ids, attention_mask, labels = to_device(list(pair), pad_id, device)
    return sequence_log_probs(model, input_ids, attention_mask, labels)

def reference_log_probs(model, pairs, pad_id, device):
    results = []
    with torch.no_grad():
        for pair in pairs:
            result = pair_log_probs(model, pair, pad_id, device)
            results.append(result.cpu())
    return torch.stack(results)

def pair_loss(model, pair, reference, beta, pad_id, device):
    policy = pair_log_probs(model, pair, pad_id, device)
    reference = reference.to(policy.device)
    return dpo_loss(policy[0], policy[1], reference[0], reference[1], beta)

def load_pairs(path, tokeniser, max_tokens):
    rows = load_examples(path)
    pairs = []
    for row in rows:
        pair = encode_pair(row, tokeniser)
        longer = max(len(pair[0][0]), len(pair[1][0]))
        if 2 * longer <= max_tokens:
            pairs.append(pair)
    print(f"{path.name}: kept {len(pairs)} of {len(rows)} pairs within {max_tokens} tokens")
    return pairs

def evaluate_pairs(model, pairs, reference, beta, pad_id, device):
    losses_total = 0.0
    number_of_preferred = 0
    number_of_pairs = len(pairs)
    model.eval()
    with torch.no_grad():
        for pair, row in zip(pairs, reference):
            policy = pair_log_probs(model, pair, pad_id, device)
            row = row.to(policy.device)
            loss = dpo_loss(policy[0], policy[1], row[0], row[1], beta)
            losses_total += loss.item()
            preferred = policy[0] > policy[1]
            number_of_preferred += int(preferred)
    model.train()
    return losses_total / number_of_pairs, number_of_preferred / number_of_pairs


def train(args):
    device = "cuda"
    tokeniser = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokeniser.pad_token_id
    train_pairs = load_pairs(args.data / "train.jsonl", tokeniser, args.max_tokens)[: args.limit]
    val_pairs = load_pairs(args.data / "val.jsonl", tokeniser, args.max_tokens)[: args.limit]
    max_steps = args.epochs * -(-len(train_pairs) // args.accumulation)

    model, trainable = load_model(args.rank, args.alpha, device, args.model, args.bits)
    load_lora_adapter(model, args.adapter / "adapter.pt")
    if args.checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    optimiser = AdamW(trainable, weight_decay=0.0)
    directory = make_directory(args.output_dir)
    print("run", directory, "pairs", len(train_pairs), "held-out pairs", len(val_pairs), "optimiser steps", max_steps)

    reference_train = reference_log_probs(model, train_pairs, pad_id, device)
    reference_val = reference_log_probs(model, val_pairs, pad_id, device)

    stats = []
    best_val = float("inf")
    step = 0
    train_sum = 0.0
    train_count = 0
    started = time.time()

    def log(lr):
        nonlocal best_val, train_sum, train_count
        val, preferred = evaluate_pairs(model, val_pairs, reference_val, args.beta, pad_id, device)
        train_mean = train_sum / train_count if train_count else float("nan")
        train_sum = 0.0
        train_count = 0
        if val < best_val:
            best_val = val
            save_lora_adapter(model, directory)
        stats.append({"step": step, "train": train_mean, "val": val, "preferred": preferred, "lr": lr})
        log_losses(stats, directory)
        print(f"step {step} train {train_mean:.4f} val {val:.4f} preferred {preferred:.3f} lr {lr:.2e}", flush=True)

    log(0.0)

    for epoch in range(args.epochs):
        order = list(range(len(train_pairs)))
        random.Random(epoch).shuffle(order)
        for start in range(0, len(order), args.accumulation):
            group = order[start : start + args.accumulation]
            for index in group:
                loss = pair_loss(model, train_pairs[index], reference_train[index], args.beta, pad_id, device)
                (loss / len(group)).backward()
                train_sum += loss.item()
                train_count += 1
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
    print(f"{elapsed:.0f} s, {elapsed / max(step * args.accumulation, 1):.1f} s per pair, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--bits", type=int, default=16, choices=(16, 4))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=10)

    train(parser.parse_args())
