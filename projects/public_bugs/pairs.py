import argparse
import random
from collections import Counter
from pathlib import Path

from finetune.dpo import pairs_from_samples
from finetune.sft import load_examples
from projects.pr_reviews.dataset import write_jsonl
from projects.pr_reviews.traces import BREAK, NO_BREAK

KEYS = ("kind", "repo", "number")
MATCH = "match"


def pair_rows(answers, examples):
    rows = []
    for answer, example in zip(answers, examples, strict=True):
        assert all(answer[key] == example[key] for key in KEYS), (answer["repo"], example["repo"])
        correct = BREAK if example["kind"] == "real" else NO_BREAK
        for chosen, rejected in pairs_from_samples(answer["samples"], correct):
            rows.append({"system": example["system"], "prompt": example["prompt"], "chosen": chosen, "rejected": rejected} | {key: example[key] for key in KEYS})
    return rows


def judged_samples(answers, verdicts):
    judged = []
    for index, answer in enumerate(answers):
        marks = verdicts.get(index, {})
        samples = [sample | {"verdict": marks.get(str(k), "none")} for k, sample in enumerate(answer["samples"])]
        judged.append(answer | {"samples": samples})
    return judged


def pair_rows_by_content(answers, examples, verdicts):
    rows = []
    for answer, example in zip(judged_samples(answers, verdicts), examples, strict=True):
        assert all(answer[key] == example[key] for key in KEYS), (answer["repo"], example["repo"])
        for chosen, rejected in pairs_from_samples(answer["samples"], MATCH):
            rows.append({"system": example["system"], "prompt": example["prompt"], "chosen": chosen, "rejected": rejected} | {key: example[key] for key in KEYS})
    return rows


def load_verdicts(paths):
    verdicts = {}
    for path in paths:
        for row in load_examples(path):
            assert row["index"] not in verdicts, row["index"]
            verdicts[row["index"]] = row["answers"]
    return verdicts


def hold_out(rows, share, seed=0):
    functions = sorted({tuple(row[key] for key in KEYS) for row in rows}, key=repr)
    held = set(random.Random(seed).sample(functions, round(len(functions) * share)))
    inside = [tuple(row[key] for key in KEYS) in held for row in rows]
    return [row for row, is_held in zip(rows, inside) if not is_held], [row for row, is_held in zip(rows, inside) if is_held]


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
    parser.add_argument("--verdicts", type=Path, nargs="+", help="reader verdict files; pairs an answer marked match with one marked no, in place of pairing by verdict")
    parser.add_argument("--val-out", type=Path, help="write the pairs of a held-out share of functions here")
    parser.add_argument("--holdout", type=float, default=0.1)
    args = parser.parse_args()

    if args.verdicts:
        rows = pair_rows_by_content(load_examples(args.answers), load_examples(args.data), load_verdicts(args.verdicts))
    else:
        rows = pair_rows(load_examples(args.answers), load_examples(args.data))
    if args.balance:
        rows = balance(rows)
    if args.val_out:
        rows, held = hold_out(rows, args.holdout)
        write_jsonl(args.val_out, held)
        print(f"{len(held)} held-out pairs", dict(Counter(row["kind"] for row in held)))
    write_jsonl(args.out, rows)
    print(f"{len(rows)} pairs", dict(Counter(row["kind"] for row in rows)))
