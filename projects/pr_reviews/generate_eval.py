import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from finetune.first_token import auc, matched_threshold, probability
from finetune.lora import load_lora_adapter
from finetune.sft import IGNORE, MODEL, encode, load_examples, load_model
from projects.pr_reviews.generator import NO_BREAK, OUTPUT

MAX_NEW_TOKENS = 200


def is_no_break(text):
    head = text.strip().strip("`").strip().lower()
    return head.startswith(NO_BREAK) or head.startswith("i cannot find a break")


def score(model, tokeniser, example, device, no_id):
    ids, labels = encode(example, tokeniser)
    input_ids = torch.tensor([ids], device=device)
    output = model(input_ids=input_ids, labels=torch.tensor([labels], device=device))
    prompt_length = labels.count(IGNORE)
    return input_ids, prompt_length, output.loss.item(), probability(output.logits[0, prompt_length - 1], no_id)


def answer(model, tokeniser, example, device, no_id, threshold):
    input_ids, prompt_length, loss, p_no = score(model, tokeniser, example, device, no_id)
    if p_no >= threshold:
        text = NO_BREAK
    else:
        generated = model.generate(
            input_ids[:, :prompt_length], max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokeniser.pad_token_id, begin_suppress_tokens=[no_id]
        )
        text = tokeniser.decode(generated[0, prompt_length:], skip_special_tokens=True)
    return {
        "kind": example["kind"],
        "repo": example["repo"],
        "number": example["number"],
        "gold": example["answer"],
        "generated": text.strip(),
        "loss": loss,
        "p_no": p_no,
    }


def run(name, model, tokeniser, examples, device, path, threshold):
    no_id = tokeniser(NO_BREAK, add_special_tokens=False).input_ids[0]
    model.eval()
    rows = []
    with torch.no_grad():
        for index, example in enumerate(examples):
            rows.append(answer(model, tokeniser, example, device, no_id, threshold))
            if index % 50 == 0:
                print(f"  {name} {index}/{len(examples)}", flush=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report(name, rows)
    return rows


def match_threshold(model, tokeniser, examples, device, base):
    no_id = tokeniser(NO_BREAK, add_special_tokens=False).input_ids[0]
    answered = sum(not is_no_break(r["generated"]) for r in base if r["kind"] == "real")
    model.eval()
    with torch.no_grad():
        scores = [score(model, tokeniser, e, device, no_id)[3] for e in examples if e["kind"] == "real"]
    threshold = matched_threshold(scores, answered)
    print(f"matched threshold {threshold:.4f} answers {answered} real examples like the base", flush=True)
    return threshold


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def report(name, rows):
    real = [r for r in rows if r["kind"] == "real"]
    clean = [r for r in rows if r["kind"] == "clean"]
    specificity = mean([is_no_break(r["generated"]) for r in clean])
    miss = mean([is_no_break(r["generated"]) for r in real])
    ranking = auc([r["p_no"] for r in clean], [r["p_no"] for r in real])
    print(
        f"{name}: real loss {mean([r['loss'] for r in real]):.3f} clean loss {mean([r['loss'] for r in clean]):.3f}"
        f" p_no real {mean([r['p_no'] for r in real]):.3f} clean {mean([r['p_no'] for r in clean]):.3f} auc {ranking:.3f}"
        f" no-break on clean {specificity:.3f} ({len(clean)}) no-break on real {miss:.3f} ({len(real)})"
        f" balanced {(specificity + 1 - miss) / 2:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=OUTPUT)
    parser.add_argument("--run")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--bits", type=int, default=16, choices=(16, 4))
    parser.add_argument("--rank", default=16, type=int)
    parser.add_argument("--alpha", default=32, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--match", action="store_true", help="pick the adapter threshold that answers as many real examples as the base did")
    args = parser.parse_args()

    device = "cuda"
    if args.run:
        directory = args.data / "runs" / args.run
    else:
        directory = args.data / "base" / args.model.split("/")[-1]
        directory.mkdir(parents=True, exist_ok=True)
    tokeniser = AutoTokenizer.from_pretrained(args.model)
    model, _ = load_model(args.rank, args.alpha, device, args.model, args.bits)
    examples = load_examples(args.data / "val.jsonl")
    if args.limit:
        examples = random.Random(0).sample(examples, args.limit)

    saved = args.data / "base" / args.model.split("/")[-1] / "answers-base.jsonl"
    if args.run and saved.exists() and not args.limit:
        base = [json.loads(line) for line in saved.read_text().splitlines()]
        report("base", base)
    else:
        base = run("base", model, tokeniser, examples, device, directory / "answers-base.jsonl", args.threshold)
    if args.run:
        load_lora_adapter(model, directory / "adapter.pt")
        threshold = match_threshold(model, tokeniser, examples, device, base) if args.match else args.threshold
        adapter = run("adapter", model, tokeniser, examples, device, directory / "answers-adapter.jsonl", threshold)
    else:
        adapter = [None] * len(base)

    real_pairs = [pair for pair in zip(base, adapter) if pair[0]["kind"] == "real"]
    for b, a in random.Random(0).sample(real_pairs, min(10, len(real_pairs))):
        print("=" * 80)
        print("gold:     " + b["gold"])
        print("base:     " + b["generated"])
        if a is not None:
            print("adapter:  " + a["generated"])
