import argparse
import json
import random
from pathlib import Path

from finetune.ollama import ask, boolean_schema
from projects.pr_reviews.generate_eval import is_no_break

RULE = (
    "You compare two code review findings on the same function: the reference and the candidate. "
    "They match when they point at the same underlying problem, even if worded differently or anchored a few lines apart. "
    "Findings that merely touch the same lines but describe different problems do not match. "
    'Reply with JSON only: {"match": true} or {"match": false}.'
)
SCHEMA = boolean_schema("match")


def match(reference, candidate):
    return ask(RULE, f"Reference finding:\n{reference[:1500]}\n\nCandidate finding:\n{candidate[:1500]}", SCHEMA)["match"]


def judge(rows):
    judged = []
    for row in rows:
        if row["kind"] != "real":
            continue
        hit = False if is_no_break(row["generated"]) else match(row["gold"], row["generated"])
        judged.append({**row, "match": hit})
    return judged


def calibrate(rows, count=10):
    real = [r for r in rows if r["kind"] == "real"]
    rng = random.Random(0)
    same = sum(match(r["gold"], r["gold"]) for r in rng.sample(real, count))
    shuffled = rng.sample(real, count)
    other = sum(match(a["gold"], b["gold"]) for a, b in zip(shuffled, shuffled[1:] + shuffled[:1]))
    print(f"calibration: same finding matched {same}/{count}, different finding matched {other}/{count}")


def judged_path(answers):
    return answers.with_name("judged-" + answers.name.removeprefix("answers-"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", type=Path)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.answers.read_text().splitlines()]
    if args.calibrate:
        calibrate(rows)
    judged = judge(rows)
    hits = sum(r["match"] for r in judged)
    answered = sum(not is_no_break(r["generated"]) for r in judged)
    print(f"{args.answers.name}: hit rate {hits / len(judged):.3f} ({hits}/{len(judged)}), answered {answered}/{len(judged)}")
    with judged_path(args.answers).open("w") as handle:
        for row in judged:
            handle.write(json.dumps(row) + "\n")
