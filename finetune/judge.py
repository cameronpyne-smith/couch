import argparse
import json
import random
import urllib.request
from pathlib import Path

from finetune.generate_eval import is_no_break

OLLAMA = "http://localhost:11434/api/chat"
JUDGE_MODEL = "qwen3.8:27b"
RULE = (
    "You compare two code review findings on the same function: the reference and the candidate. "
    "They match when they point at the same underlying problem, even if worded differently or anchored a few lines apart. "
    "Findings that merely touch the same lines but describe different problems do not match. "
    'Reply with JSON only: {"match": true} or {"match": false}.'
)
SCHEMA = {"type": "object", "properties": {"match": {"type": "boolean"}}, "required": ["match"]}


def ask(reference, candidate):
    body = {
        "model": JUDGE_MODEL,
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": RULE},
            {"role": "user", "content": f"Reference finding:\n{reference[:1500]}\n\nCandidate finding:\n{candidate[:1500]}"},
        ],
    }
    request = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        reply = json.load(response)
    return json.loads(reply["message"]["content"])["match"]


def judge(rows):
    judged = []
    for row in rows:
        if row["kind"] != "real":
            continue
        match = False if is_no_break(row["generated"]) else ask(row["gold"], row["generated"])
        judged.append({**row, "match": match})
    return judged


def calibrate(rows, count=10):
    real = [r for r in rows if r["kind"] == "real"]
    rng = random.Random(0)
    same = sum(ask(r["gold"], r["gold"]) for r in rng.sample(real, count))
    shuffled = rng.sample(real, count)
    other = sum(ask(a["gold"], b["gold"]) for a, b in zip(shuffled, shuffled[1:] + shuffled[:1]))
    print(f"calibration: same finding matched {same}/{count}, different finding matched {other}/{count}")


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
    out = args.answers.with_name(args.answers.name.replace("answers", "judged"))
    with out.open("w") as handle:
        for row in judged:
            handle.write(json.dumps(row) + "\n")
