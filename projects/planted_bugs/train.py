import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune.lora import apply_lora, save_lora_adapter
from finetune.sft import MODEL, PROJECTIONS
from optimiser import AdamW, clip_gradients
from train import get_learning_rate

IGNORE = -100
METRICS = "windlet-training/1 "
SYSTEM = (
    "You review code. One line of the function below may contain a small bug: an inverted or wrong comparison, "
    "a wrong logical operator, a flipped boolean, a dropped negation, a dropped null check or an off-by-one. "
    "Answer with `Line: N`, the number of the line with the bug, or `Line: none` when the function has no bug."
)
NUMBER = re.compile(r"\s*(\d+)")


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]


def prompt_ids(row, tokeniser):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["code"]}]
    head = tokeniser.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tokeniser(head + "Line:", add_special_tokens=False).input_ids


def answer_ids(row, tokeniser):
    text = f" {row['bug_line']}" if row["bug_line"] else " none"
    return tokeniser(text, add_special_tokens=False).input_ids + [tokeniser.eos_token_id]


def encode(row, tokeniser):
    prompt = prompt_ids(row, tokeniser)
    return prompt + answer_ids(row, tokeniser), len(prompt)


def make_batches(items, max_tokens, rng):
    batches = [[]]
    for item in sorted(items, key=lambda item: len(item[0])):
        if batches[-1] and (len(batches[-1]) + 1) * len(item[0]) > max_tokens:
            batches.append([])
        batches[-1].append(item)
    rng.shuffle(batches)
    return batches


def left_padded(sequences, pad_id):
    width = max(len(s) for s in sequences)
    ids = torch.tensor([[pad_id] * (width - len(s)) + s for s in sequences], device="cuda")
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in sequences], device="cuda")
    return ids, mask


def answer_logits(model, batch, pad_id):
    ids, mask = left_padded([ids for ids, _ in batch], pad_id)
    labels = torch.full_like(ids, IGNORE)
    width = ids.shape[1]
    for k, (sequence, prompt_length) in enumerate(batch):
        start = width - len(sequence) + prompt_length
        labels[k, start:] = ids[k, start:]
    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
    targets = labels[:, 1:]
    keep = targets != IGNORE
    return model.lm_head(hidden[keep]).float(), targets[keep], keep


def auc(scores, labels):
    positives = [s for s, l in zip(scores, labels) if l]
    negatives = [s for s, l in zip(scores, labels) if not l]
    return sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives) / (len(positives) * len(negatives))


@torch.no_grad()
def answers(model, tokeniser, rows, pad_id, max_tokens):
    said = {}
    for batch in make_batches([(prompt_ids(r, tokeniser), i) for i, r in enumerate(rows)], max_tokens, random.Random(0)):
        ids, mask = left_padded([ids for ids, _ in batch], pad_id)
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=6, do_sample=False, pad_token_id=pad_id)
        for k, (_, i) in enumerate(batch):
            said[i] = tokeniser.decode(out[k, ids.shape[1] :], skip_special_tokens=True).strip()
    return [said[i] for i in range(len(rows))]


def named_line(answer):
    match = NUMBER.match(answer)
    return int(match.group(1)) if match else None


@torch.no_grad()
def evaluate(model, tokeniser, val, pad_id, none_id, max_tokens):
    model.eval()
    model.config.use_cache = True
    loss_sum = tokens = 0
    p_bug = {}
    for batch in make_batches([(*encode(r, tokeniser), i) for i, r in enumerate(val)], max_tokens, random.Random(0)):
        logits, targets, keep = answer_logits(model, [(ids, length) for ids, length, _ in batch], pad_id)
        loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        tokens += len(targets)
        first = keep.float().argmax(dim=1)
        rows_of = keep.sum(dim=1).cumsum(0) - keep.sum(dim=1)
        probabilities = F.softmax(logits[rows_of], dim=-1)[:, none_id]
        for k, (_, _, i) in enumerate(batch):
            p_bug[i] = 1.0 - probabilities[k].item()
    said = answers(model, tokeniser, val, pad_id, max_tokens)
    model.config.use_cache = False
    model.train()
    planted = [i for i, r in enumerate(val) if r["bug_line"]]
    clean = [i for i, r in enumerate(val) if not r["bug_line"]]
    pairs = defaultdict(dict)
    for i, r in enumerate(val):
        pairs[r["pair"]]["planted" if r["bug_line"] else "clean"] = p_bug[i]
    return {
        "val_loss": loss_sum / tokens,
        "val_found": sum(named_line(said[i]) == val[i]["bug_line"] for i in planted) / len(planted),
        "val_false_alarm": sum(named_line(said[i]) is not None for i in clean) / len(clean),
        "val_pair_ranked": sum((p["planted"] > p["clean"]) + 0.5 * (p["planted"] == p["clean"]) for p in pairs.values()) / len(pairs),
        "val_auc": auc([p_bug[i] for i in range(len(val))], [bool(r["bug_line"]) for r in val]),
    }, said


def pick_samples(val, count):
    by_repo = defaultdict(list)
    for r in val:
        if r["bug_line"]:
            by_repo[r["repo"]].append(r)
    chosen, mutations = [], set()
    while len(chosen) < count and any(by_repo.values()):
        for rows in by_repo.values():
            fresh = next((r for r in rows if r["mutation"] not in mutations), rows[0] if rows else None)
            if fresh is None or len(chosen) >= count:
                continue
            rows.remove(fresh)
            chosen.append(fresh["pair"])
            mutations.add(fresh["mutation"])
    return [i for i, r in enumerate(val) if r["pair"] in chosen]


def metrics(step, values):
    print(METRICS + json.dumps({"step": step, **{k: round(v, 5) for k, v in values.items()}}), flush=True)


def train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    started = time.time()
    tokeniser = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokeniser.pad_token_id if tokeniser.pad_token_id is not None else tokeniser.eos_token_id
    none_id = tokeniser(" none", add_special_tokens=False).input_ids[0]
    train_rows, val = load(args.input_data_path)
    encoded = [encode(r, tokeniser) for r in train_rows]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    trainable = apply_lora(model, args.rank, 2 * args.rank, PROJECTIONS)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    optimiser = AdamW(trainable, weight_decay=0.0)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    eval_interval = args.eval_interval or max(10, args.steps // 10)
    samples = pick_samples(val, args.samples)
    print(f"{len(train_rows)} training and {len(val)} validation functions from {len({r['repo'] for r in train_rows + val})} repositories, loaded in {time.time() - started:.0f} s", flush=True)

    result, said = evaluate(model, tokeniser, val, pad_id, none_id, args.max_tokens)
    metrics(0, result)
    before = {i: said[i] for i in samples}
    best = result["val_loss"]
    step = 0
    loss_sum = tokens = 0
    while step < args.steps:
        for batch in make_batches(encoded, args.max_tokens, rng):
            lr = get_learning_rate(step, warm_up_steps=min(20, args.steps // 5 + 1), max_steps=args.steps, max_learning_rate=args.lr, min_learning_rate=args.lr / 10)
            logits, targets, _ = answer_logits(model, batch, pad_id)
            loss = F.cross_entropy(logits, targets, reduction="sum")
            (loss / len(targets)).backward()
            clip_gradients(trainable, 1.0)
            optimiser.learning_rate = lr
            optimiser.step()
            optimiser.zero_grad()
            step += 1
            loss_sum += loss.item()
            tokens += len(targets)
            if step % 10 == 0 or step == args.steps:
                metrics(step, {"loss": loss_sum / tokens, "lr": lr})
                loss_sum = tokens = 0
            if step % eval_interval == 0 or step == args.steps:
                result, said = evaluate(model, tokeniser, val, pad_id, none_id, args.max_tokens)
                metrics(step, result)
                if result["val_loss"] < best:
                    best = result["val_loss"]
                    save_lora_adapter(model, output)
            if step >= args.steps:
                break

    rows = [{"repo": val[i]["repo"], "path": val[i]["path"], "licence": val[i]["licence"], "code": val[i]["code"], "bug_line": val[i]["bug_line"], "mutation": val[i]["mutation"], "before": before[i], "after": said[i]} for i in samples]
    (output / "samples.json").write_text(json.dumps(rows, indent=2))
    (output / "result.json").write_text(json.dumps({"model": args.model, "steps": step, "lr": args.lr, "rank": args.rank, "seconds": round(time.time() - started), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), **result}, indent=2))
    for row in rows:
        truth = row["bug_line"] or "none"
        print(f"\n{row['repo']} {row['path']} ({row['mutation'] or 'no bug'}): answer {truth}, before {row['before']!r}, after {row['after']!r}")
    print(f"\n{step} steps in {time.time() - started:.0f} s, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-path", default=str(Path(__file__).parent / "data" / "planted-bugs.jsonl"))
    parser.add_argument("--output-dir", default="./runs/planted-bugs")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=24576)
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())
