import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from finetune.eval import load_adapter
from finetune.generator import NO_BREAK
from finetune.sft import IGNORE, MODEL, encode, load_examples, load_model

MAX_NEW_TOKENS = 200


def is_no_break(text):
    return text.strip().lower().startswith(NO_BREAK)


def answer(model, tokeniser, example, device, no_id, threshold):
    ids, labels = encode(example, tokeniser)
    input_ids = torch.tensor([ids], device=device)
    output = model(input_ids=input_ids, labels=torch.tensor([labels], device=device))
    prompt_length = labels.count(IGNORE)
    p_no = torch.softmax(output.logits[0, prompt_length - 1].float(), dim=-1)[no_id].item()
    prompt = input_ids[:, :prompt_length]
    suppress = [] if p_no >= threshold else [no_id]
    generated = model.generate(
        prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokeniser.pad_token_id, begin_suppress_tokens=suppress
    )
    text = tokeniser.decode(generated[0, prompt_length:], skip_special_tokens=True)
    return {
        "kind": example["kind"],
        "repo": example["repo"],
        "number": example["number"],
        "gold": example["answer"],
        "generated": text.strip(),
        "loss": output.loss.item(),
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
                print(f"  {name} {index}/{len(examples)}")
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report(name, rows)
    return rows


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def ranking(real, clean):
    if not real or not clean:
        return float("nan")
    return sum(r["p_no"] < c["p_no"] for r in real for c in clean) / (len(real) * len(clean))


def report(name, rows):
    real = [r for r in rows if r["kind"] == "real"]
    clean = [r for r in rows if r["kind"] == "clean"]
    specificity = mean([is_no_break(r["generated"]) for r in clean])
    miss = mean([is_no_break(r["generated"]) for r in real])
    print(
        f"{name}: real loss {mean([r['loss'] for r in real]):.3f} clean loss {mean([r['loss'] for r in clean]):.3f}"
        f" p_no real {mean([r['p_no'] for r in real]):.3f} clean {mean([r['p_no'] for r in clean]):.3f} auc {ranking(real, clean):.3f}"
        f" no-break on clean {specificity:.3f} ({len(clean)}) no-break on real {miss:.3f} ({len(real)})"
        f" balanced {(specificity + 1 - miss) / 2:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path.home() / "training-data" / "code" / "pr-reviews-generator")
    parser.add_argument("--run", default="20261106-121710")
    parser.add_argument("--rank", default=16, type=int)
    parser.add_argument("--alpha", default=32, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda"
    run_dir = args.data / "runs" / args.run
    tokeniser = AutoTokenizer.from_pretrained(MODEL)
    model, _ = load_model(args.rank, args.alpha, device)
    examples = load_examples(args.data / "val.jsonl")
    if args.limit:
        examples = random.Random(0).sample(examples, args.limit)

    base = run("base", model, tokeniser, examples, device, run_dir / "answers-base.jsonl", args.threshold)
    load_adapter(model, run_dir / "adapter.pt")
    adapter = run("adapter", model, tokeniser, examples, device, run_dir / "answers-adapter.jsonl", args.threshold)

    real_pairs = [pair for pair in zip(base, adapter) if pair[0]["kind"] == "real"]
    for b, a in random.Random(0).sample(real_pairs, min(10, len(real_pairs))):
        print("=" * 80)
        print("gold:     " + b["gold"])
        print("base:     " + b["generated"])
        print("adapter:  " + a["generated"])
