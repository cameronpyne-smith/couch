import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from finetune.sampling import any_of_k, sample
from finetune.sft import IGNORE, encode, load_base, load_examples
from projects.pr_reviews.context import SUBSET
from projects.pr_reviews.generate_eval import MAX_NEW_TOKENS
from projects.pr_reviews.generator import NO_BREAK


def summarise(judged_path):
    rows = [json.loads(line) for line in judged_path.read_text().splitlines()]
    any_of_k(rows, key=lambda r: (r["repo"], r["number"], r["gold"]), hit=lambda r: r["match"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=SUBSET)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--summarise", type=Path)
    args = parser.parse_args()

    if args.summarise:
        summarise(args.summarise)
        raise SystemExit

    device = "cuda"
    tokeniser = AutoTokenizer.from_pretrained(args.model)
    model = load_base(args.model, 4, device)
    model.eval()
    no_id = tokeniser(NO_BREAK, add_special_tokens=False).input_ids[0]
    examples = [e for e in load_examples(args.data / "val.jsonl") if e["kind"] == "real"]
    out = args.data / "base" / args.model.split("/")[-1] / f"answers-samples-k{args.count}.jsonl"
    with out.open("w") as handle, torch.no_grad():
        for index, example in enumerate(examples):
            ids, labels = encode(example, tokeniser)
            prompt_ids = ids[: labels.count(IGNORE)]
            for position, text in enumerate(sample(model, tokeniser, prompt_ids, args.count, args.temperature, MAX_NEW_TOKENS, suppress=[no_id])):
                row = {"kind": "real", "repo": example["repo"], "number": example["number"], "gold": example["answer"], "sample": position, "generated": text}
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            if index % 10 == 0:
                print(f"  {index + 1}/{len(examples)}", flush=True)
    print("written", out, flush=True)
