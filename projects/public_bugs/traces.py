import argparse
import json
import random
import re
from collections import Counter

from projects.pr_reviews.dataset import write_jsonl
from projects.pr_reviews.traces import BREAK, NO_BREAK, SYSTEM, compose
from projects.public_bugs.dataset import DATA, load

PACKETS = DATA / "packets"
EXCLUDED = re.compile(r"(^|/)(www|website|scripts|deno|examples?|docs?|playground)/|\.config\.[cm]?[jt]s$|\.d\.ts$|Tests?\.Unit|CompileTest|/[^/]*[Tt]ests?/[^/]*Test", re.IGNORECASE)


def excluded(example):
    return EXCLUDED.search(example["path"]) is not None


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
        lines += ["## The fix that landed later: the defect your trace must reach", "", example["answer"], ""]
    else:
        lines += ["## Label", "", "No later fix touched this file. Your trace must reach `no break` unless the shown code has a genuine defect.", ""]
    return "\n".join(lines)


def write_packets():
    for split in ("train", "val"):
        directory = PACKETS / split
        directory.mkdir(parents=True, exist_ok=True)
        examples = load(DATA / f"{split}.jsonl")
        for index, example in enumerate(examples):
            (directory / f"{index:04d}.md").write_text(packet(split, index, example))
        print(split, dict(Counter(e["kind"] for e in examples)))


def assignments(size):
    (DATA / "assignments").mkdir(exist_ok=True)
    for split in ("train", "val"):
        examples = load(DATA / f"{split}.jsonl")
        for kind in ("real", "clean"):
            indices = [i for i, e in enumerate(examples) if e["kind"] == kind]
            for n, offset in enumerate(range(0, len(indices), size), start=1):
                path = DATA / "assignments" / f"{split}-{kind}-{n}.txt"
                path.write_text("".join(f"{PACKETS / split / f'{i:04d}.md'}\n" for i in indices[offset : offset + size]))
                print(path.name, len(indices[offset : offset + size]))


def latest(pattern):
    rows = {}
    for path in sorted(DATA.glob(pattern)):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(row["split"], int(row["index"]))] = {**row, "source": path.name}
    return rows


def check_assignments(size):
    assigned = {(row["split"], int(row["index"])) for path in (DATA / "assignments").glob("check-*.jsonl") for row in map(json.loads, path.read_text().splitlines())}
    pending = [row for key, row in sorted(latest("traces-*.jsonl").items()) if key not in assigned]
    existing = len(list((DATA / "assignments").glob("check-*.jsonl")))
    for offset in range(0, len(pending), size):
        existing += 1
        path = DATA / "assignments" / f"check-{existing}.jsonl"
        write_jsonl(path, [{"packet": str(PACKETS / row["split"] / f"{int(row['index']):04d}.md"), **row} for row in pending[offset : offset + size]])
        print(path.name, len(pending[offset : offset + size]))


def assemble():
    traces = latest("traces-*.jsonl")
    checks = latest("checks-*.jsonl")
    for split in ("train", "val"):
        examples = load(DATA / f"{split}.jsonl")
        kept, full = [], []
        count = Counter()
        for index, example in enumerate(examples):
            trace = traces.get((split, index))
            check = checks.get((split, index))
            row = {**example, "gold": example["answer"]}
            if excluded(example):
                count[(example["kind"], "excluded path")] += 1
                continue
            if trace and check and check["ok"] and trace.get("reachable", True) and not trace.get("disagree") and trace["verdict"] == (BREAK if example["kind"] == "real" else NO_BREAK):
                row["answer"] = compose(trace)
                kept.append(row)
                count[(example["kind"], "kept")] += 1
            else:
                count[(example["kind"], "no trace" if not trace else "unchecked" if not check else "rejected" if not check["ok"] else "unreachable" if not trace.get("reachable", True) else "disagree" if trace.get("disagree") else "wrong verdict")] += 1
            full.append(row)
        write_jsonl(DATA / f"{split}-traces.jsonl", kept)
        write_jsonl(DATA / f"{split}-full.jsonl", full)
        print(split, dict(sorted(count.items())))


def sft_split(holdout, real_share, seed):
    rows = load(DATA / "train-traces.jsonl")
    rng = random.Random(seed)
    reals = [r for r in rows if r["kind"] == "real"]
    cleans = [r for r in rows if r["kind"] == "clean"]
    rng.shuffle(cleans)
    cleans = cleans[: min(len(cleans), round(len(reals) * (1 - real_share) / real_share))]
    rows = reals + cleans
    rng.shuffle(rows)
    cut = round(len(rows) * holdout)
    (DATA / "sft").mkdir(exist_ok=True)
    write_jsonl(DATA / "sft" / "val.jsonl", rows[:cut])
    write_jsonl(DATA / "sft" / "train.jsonl", rows[cut:])
    print("sft train", dict(Counter(r["kind"] for r in rows[cut:])), "val", dict(Counter(r["kind"] for r in rows[:cut])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("packets", "assign", "checks", "assemble", "sft"))
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--real-share", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "packets":
        write_packets()
    elif args.command == "assign":
        assignments(args.size)
    elif args.command == "checks":
        check_assignments(args.size)
    elif args.command == "sft":
        sft_split(args.holdout, args.real_share, args.seed)
    else:
        assemble()
