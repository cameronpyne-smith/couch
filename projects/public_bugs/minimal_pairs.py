import argparse
import re
import statistics
import subprocess

from projects.pr_reviews.context import HUNK, nearest, with_diff
from projects.pr_reviews.dataset import render, write_jsonl
from projects.pr_reviews.generator import frame
from projects.pr_reviews.traces import NO_BREAK
from projects.public_bugs.dataset import DATA, MAX_PROMPT_CHARS, changed_lines, git, hunks_of, load, locate

PAIRS = DATA / "minimal-pairs"
KEPT = "pair"
OLD = re.compile(r"^@@ -(\d+)(?:,(\d+))?")


def hunks_between(repo, old, new, path):
    try:
        diff = git(repo, "diff", old, new, "--", path)
    except subprocess.CalledProcessError:
        return []
    return ["@@ " + hunk.rstrip("\n") for hunk in diff.split("\n@@ ")[1:]]


def fix_lines(hunks):
    return changed_lines(hunks) or [int(HUNK.match(hunk).group(1)) for hunk in hunks]


def old_lines(hunks):
    lines = []
    for hunk in hunks:
        match = OLD.match(hunk)
        start = int(match.group(1))
        lines.extend(range(start, start + max(int(match.group(2) or 1), 1)))
    return lines


def signature(unit):
    return unit.split("\n")[0].strip()


def half(row, commit, located, kind, answer):
    text, span, line = located
    unit = render(text, span, numbered=False)
    hunks = hunks_between(row["repo"], f"{row['commit']}^", commit, row["path"])
    if not hunks:
        return None
    prompt = with_diff(frame(unit, span[0], row["path"], f"{row['repo']}#{row['fix_pr']}"), nearest(hunks, line))
    return {
        "system": row["system"],
        "prompt": prompt,
        "answer": answer,
        "kind": kind,
        "repo": row["repo"],
        "number": row["fix_pr"],
        "commit": commit,
        "path": row["path"],
        "line": line,
        "licence": row.get("licence"),
    }


def build_pair(row):
    hunks = hunks_of(row["repo"], row["fix_commit"], row["path"])
    if not hunks:
        return None, "fix does not touch the file"
    parent = git(row["repo"], "rev-parse", f"{row['fix_commit']}^").strip()
    before = locate(row["repo"], parent, row["path"], old_lines(hunks))
    after = locate(row["repo"], row["fix_commit"], row["path"], fix_lines(hunks))
    if before is None or after is None:
        return None, "no function on one side of the fix"
    units = [render(text, span, numbered=False) for text, span, _ in (before, after)]
    if signature(units[0]) != signature(units[1]):
        return None, "another function or a changed signature"
    if units[0] == units[1]:
        return None, "function unchanged by the fix"
    buggy = half(row, parent, before, "real", row["answer"])
    fixed = half(row, row["fix_commit"], after, "clean", NO_BREAK)
    if buggy is None or fixed is None:
        return None, "no diff from before the bug"
    if max(len(buggy["prompt"]), len(fixed["prompt"])) > MAX_PROMPT_CHARS:
        return None, "too long"
    return (buggy, fixed), KEPT


def build(split):
    reasons = {}
    rows = []
    for row in load(DATA / f"{split}.jsonl"):
        if row["kind"] != "real":
            continue
        halves, reason = build_pair(row)
        reasons[reason] = reasons.get(reason, 0) + 1
        if halves is not None:
            rows.extend({**example, "pair": len(rows) // 2} for example in halves)
    write_jsonl(PAIRS / f"{split}.jsonl", rows)
    print(split, reasons)
    describe(rows)


def parts(prompt):
    function, _, diff = prompt.partition("\n\n### Changes to this file in the pull request\n")
    return len(function), len(diff)


def describe(rows):
    buggy = [parts(row["prompt"]) for row in rows[0::2]]
    fixed = [parts(row["prompt"]) for row in rows[1::2]]
    for index, name in enumerate(("function", "diff")):
        longer = sum(b[index] > f[index] for b, f in zip(buggy, fixed))
        equal = sum(b[index] == f[index] for b, f in zip(buggy, fixed))
        print(
            f"  {name} characters, median - buggy: {statistics.median(b[index] for b in buggy):.0f} - fixed: {statistics.median(f[index] for f in fixed):.0f}"
            f" - pairs where buggy is longer: {longer} - equal: {equal} - shorter: {len(buggy) - longer - equal}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "val-full"])
    args = parser.parse_args()
    for split in args.splits:
        build(split)
