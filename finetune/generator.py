import random
import re
from collections import Counter
from pathlib import Path

from finetune.dataset import LINE_REFERENCE, load_records, locate, render, write_jsonl

OUTPUT = Path.home() / "training-data" / "code" / "pr-reviews-generator"
NO_BREAK = "no break found"
UNIT_PROMPT = (
    "For each function, first state its contract in two sentences: what it promises its callers and the rest of the system. "
    "Then find one concrete sequence of events under which the function breaks that contract. Give the line where it goes wrong.\n\n"
    "Severity: critical means it will certainly break, lose data or open a security hole; major means a likely bug; "
    "minor means worth fixing but not blocking; nit means optional."
)
OUTSIDE_REFERENCE = re.compile(r"#\d+|\bPR\b|pull request|\.(cs|ts|svelte|json|ya?ml)\b|commit", re.IGNORECASE)


def frame(record):
    finding = record["finding"]
    style = random.Random(finding["id"])
    numbered = style.random() < 0.5 or LINE_REFERENCE.search(finding["body"]) is not None
    unit = render(record["text"], record["span"], numbered=numbered)
    header = f"### {finding['path']}\n" if style.random() < 0.5 else ""
    return header + unit


def build_real(record):
    finding = record["finding"]
    return {
        "system": UNIT_PROMPT,
        "prompt": frame(record),
        "answer": finding["body"],
        "kind": "real",
        "labelled": finding["labelled"],
        "repo": record["repo"],
        "number": record["number"],
        "reviewed_at": record["reviewed_at"],
    }


if __name__ == "__main__":
    records = [r for r in load_records() if r["finding"]["verdict"] == "valid"]
    print("valid findings", len(records), "located", Counter(locate(r) for r in records))
    examples = [build_real(r) for r in records if "span" in r]

    lengths = sorted(len(e["answer"]) for e in examples)
    print("answer chars median", lengths[len(lengths) // 2], "p90", lengths[int(len(lengths) * 0.9)], "max", lengths[-1])
    print("answers mentioning something outside the function", sum(OUTSIDE_REFERENCE.search(e["answer"]) is not None for e in examples))

    for e in random.Random(0).sample(examples, 20):
        print("=" * 80)
        print(e["prompt"])
        print("-> " + e["answer"])
        print("   ", e["labelled"], e["repo"], e["number"])

    write_jsonl(OUTPUT / "real.jsonl", examples)
    print("wrote", len(examples), "to", OUTPUT / "real.jsonl")
