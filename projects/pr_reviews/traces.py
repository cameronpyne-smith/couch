import argparse
import json
from collections import Counter

from projects.pr_reviews.contained import load
from projects.pr_reviews.dataset import write_jsonl
from projects.pr_reviews.verified import VERIFIED

TRACES = VERIFIED.with_name(VERIFIED.name.replace("-verified", "-traces"))
PACKETS = TRACES / "packets"
VERDICT = "Verdict:"
BREAK = "break"
NO_BREAK = "no break"
SYSTEM = (
    "You are reviewing one function from a pull request. The function is shown first, then the pull request's changes to its file as a unified diff. "
    "First reason in a few sentences: what the change did, what the function promises its callers and the rest of the system, and whether some concrete sequence of events breaks that promise. "
    f"Then write a line reading exactly `{VERDICT} {BREAK}` or `{VERDICT} {NO_BREAK}`. "
    f"After `{VERDICT} {BREAK}`, state the break in one to three sentences and quote the code where it goes wrong."
)


GENERATED = (".svelte-kit/output", "create_ssr_component")


def generated(example):
    return any(marker in example["prompt"] for marker in GENERATED)


def wanted(example):
    return example["kind"] == "clean" or example["contained"]


def packet(split, index, example):
    lines = [
        f"# {split} {index:04d} {example['kind']}",
        "",
        "## System prompt the student will see",
        "",
        SYSTEM,
        "",
        "## User prompt the student will see",
        "",
        "````",
        example["prompt"],
        "````",
        "",
    ]
    if example["kind"] == "real":
        lines += ["## The reviewer's finding: the defect your trace must reach", "", example["answer"], ""]
    else:
        lines += ["## Label", "", "A reader with repository access found no defect in this function. Your trace must reach `no break`.", ""]
    return "\n".join(lines)


def write_packets():
    for split in ("train", "val"):
        examples = load(VERIFIED / f"{split}.jsonl")
        directory = PACKETS / split
        directory.mkdir(parents=True, exist_ok=True)
        count = Counter()
        for index, example in enumerate(examples):
            if wanted(example):
                (directory / f"{index:04d}.md").write_text(packet(split, index, example))
                count[example["kind"]] += 1
        print(split, dict(count))


def compose(row):
    text = f"{row['trace'].strip()}\n\n{VERDICT} {row['verdict']}"
    if row["verdict"] == BREAK:
        text += f"\n\n{row['finding'].strip()}"
    return text


def latest(pattern):
    rows = {}
    for path in sorted(TRACES.glob(pattern)):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(row["split"], int(row["index"]))] = {**row, "source": path.name}
    return rows


def check_assignments(size):
    assigned = {(row["split"], int(row["index"])) for path in (TRACES / "assignments").glob("check-*.jsonl") for row in map(json.loads, path.read_text().splitlines())}
    pending = [row for key, row in sorted(latest("traces-*.jsonl").items()) if key not in assigned]
    existing = len(list((TRACES / "assignments").glob("check-*.jsonl")))
    for offset in range(0, len(pending), size):
        existing += 1
        path = TRACES / "assignments" / f"check-{existing}.jsonl"
        rows = [{"packet": str(PACKETS / row["split"] / f"{int(row['index']):04d}.md"), **row} for row in pending[offset : offset + size]]
        write_jsonl(path, rows)
        print(path.name, len(rows))


def assemble():
    traces = latest("traces-*.jsonl")
    checks = latest("checks-*.jsonl")
    for split in ("train", "val"):
        examples = load(VERIFIED / f"{split}.jsonl")
        kept, full = [], []
        count = Counter()
        for index, example in enumerate(examples):
            trace = traces.get((split, index))
            check = checks.get((split, index))
            row = {**example, "system": SYSTEM, "gold": example["answer"]}
            if generated(example):
                count[(example["kind"], "generated")] += 1
                continue
            if trace and check and check["ok"] and trace["verdict"] == (BREAK if example["kind"] == "real" else NO_BREAK):
                row["answer"] = compose(trace)
                kept.append(row)
                count[(example["kind"], "kept")] += 1
            else:
                count[(example["kind"], "no trace" if not trace else "unchecked" if not check else "rejected" if not check["ok"] else "wrong verdict")] += 1
            full.append(row)
        write_jsonl(TRACES / f"{split}.jsonl", kept)
        if split == "val":
            write_jsonl(TRACES / "val-full.jsonl", full)
        print(split, dict(sorted(count.items())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("packets", "checks", "assemble"))
    parser.add_argument("--size", type=int, default=65)
    args = parser.parse_args()
    if args.command == "packets":
        write_packets()
    elif args.command == "checks":
        check_assignments(args.size)
    else:
        assemble()
