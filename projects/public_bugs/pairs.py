import argparse
import random
from collections import Counter
from pathlib import Path

from finetune.dpo import pairs_from_samples
from finetune.sft import load_examples
from projects.pr_reviews.dataset import write_jsonl
from projects.pr_reviews.traces import BREAK, NO_BREAK

KEYS = ("kind", "repo", "number")


def pair_rows(answers, examples):
    rows = []
    for answer, example in zip(answers, examples, strict=True):
        assert all(answer[key] == example[key] for key in KEYS), (answer["repo"], example["repo"])
        correct = BREAK if example["kind"] == "real" else NO_BREAK
        for chosen, rejected in pairs_from_samples(answer["samples"], correct):
            rows.append({"system": example["system"], "prompt": example["prompt"], "chosen": chosen, "rejected": rejected} | {key: example[key] for key in KEYS})
    return rows


def balance(rows, seed=0):
    by_kind = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row)
    size = min(len(group) for group in by_kind.values())
    kept = {id(row) for group in by_kind.values() for row in random.Random(seed).sample(group, size)}
    return [row for row in rows if id(row) in kept]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--balance", action="store_true", help="keep the same number of pairs for each kind")
    args = parser.parse_args()

    rows = pair_rows(load_examples(args.answers), load_examples(args.data))
    if args.balance:
        rows = balance(rows)
    write_jsonl(args.out, rows)
    print(f"{len(rows)} pairs", dict(Counter(row["kind"] for row in rows)))
