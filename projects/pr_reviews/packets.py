import argparse
import json
import re
from pathlib import Path

from projects.pr_reviews.contained import load
from projects.pr_reviews.context import SUBSET
from projects.pr_reviews.dataset import CLONES, GOLDEN
from projects.pr_reviews.generator import OUTPUT

REVIEW = OUTPUT.with_name("pr-reviews-review")
PUBLIC_CLONES = Path.home() / "training-data" / "code" / "public-bugs" / "clones"


def findings_by_pr():
    index = {}
    for golden_path in GOLDEN.rglob("*.json"):
        golden = json.loads(golden_path.read_text())
        for finding in golden.get("findings") or []:
            index.setdefault((golden["repo"], golden["number"]), []).append(finding)
    return index


def pool_by_pr():
    index = {}
    for function in load(OUTPUT / "pool.jsonl"):
        index.setdefault((function["repo"], function["number"]), []).append(function)
    return index


NUMBERED = re.compile(r"^\s*\d+   ", re.MULTILINE)


def stripped(text):
    return "".join(NUMBERED.sub("", text).split())


def gold(example):
    return example.get("gold", example["answer"])


def clone_for(repo):
    public = PUBLIC_CLONES / repo.replace("/", "__")
    return public if public.exists() else CLONES / repo.split("/", 1)[1]


def locate(example, findings, pool):
    if {"commit", "path", "line"} <= example.keys():
        return {"commit": example["commit"], "path": example["path"], "line": example["line"]}
    pr = (example["repo"], example["number"])
    if example["kind"] == "real":
        for finding in findings.get(pr, []):
            if finding["body"] == gold(example):
                return {"commit": finding["commit"], "path": finding["path"], "line": finding["line"]}
        return {}
    body = stripped(example["prompt"].split("### Changes to this file")[0])
    for function in pool.get(pr, []):
        if stripped(function["unit"]) in body or stripped(function["unit"]) == body.split("###", 1)[-1]:
            return {"commit": function["commit"], "path": function["path"], "line": function["span"][0] + 1}
    for function in pool.get(pr, []):
        if stripped("\n".join(function["unit"].split("\n")[:3])) in body:
            return {"commit": function["commit"], "path": function["path"], "line": function["span"][0] + 1}
    return {}


def packet(index, example, where, answers):
    clone = clone_for(example["repo"])
    lines = [
        f"# {index:04d} {example['kind']} {example['repo']}#{example['number']}",
        "",
        f"- repo clone: `{clone}`",
        f"- commit: `{where.get('commit', 'unknown')}`",
        f"- file: `{where.get('path', 'unknown')}` line {where.get('line', '?')}",
        f"- read other files at this commit with: `git -C {clone} show {where.get('commit', '<commit>')}:<path>`",
        f"- list files in the PR diff: `git -C {clone} diff --stat {where.get('commit', '<commit>')}~1 {where.get('commit', '<commit>')}`" if where.get("commit") else "",
        "",
        "## Prompt the models saw",
        "",
        "System: " + example["system"],
        "",
        "````",
        example["prompt"],
        "````",
        "",
        "## Gold answer (the reviewer or the later fix, or `no break found` for clean)",
        "",
        gold(example),
        "",
    ]
    for name, row in answers.items():
        lines += [f"## Finding by {name} (p_refuse {row['p_no']:.3f}" + (f", verdict {row['verdict']}" if row.get("verdict") else "") + ")", "", row["generated"], ""]
    return "\n".join(line for line in lines if line is not None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=SUBSET)
    parser.add_argument("--split", default="val")
    parser.add_argument("--kind", choices=("real", "clean"))
    parser.add_argument("--answers", nargs="+", default=[], help="name=path pairs of answer files aligned with the split")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    out = args.out or REVIEW / args.data.name / args.split
    out.mkdir(parents=True, exist_ok=True)
    examples = load(args.data / f"{args.split}.jsonl")
    sources = {}
    for pair in args.answers:
        name, path = pair.split("=", 1)
        rows = load(Path(path))
        expected = sum(e["kind"] == args.kind for e in examples) if args.kind else len(examples)
        assert len(rows) <= expected, (name, len(rows), expected)
        sources[name] = iter(rows)
    findings = findings_by_pr()
    pool = pool_by_pr()
    manifest = []
    missing = 0
    for index, example in enumerate(examples):
        if args.kind and example["kind"] != args.kind:
            continue
        try:
            answers = {name: next(rows) for name, rows in sources.items()}
        except StopIteration:
            break
        for row in answers.values():
            assert row["gold"] == gold(example) and row["number"] == example["number"], index
        where = locate(example, findings, pool)
        missing += not where
        (out / f"{index:04d}.md").write_text(packet(index, example, where, answers))
        manifest.append({"index": index, "kind": example["kind"], "repo": example["repo"], "number": example["number"], **where, "answers": list(answers)})
    manifest_path = out / "manifest.jsonl"
    merged = {row["index"]: row for row in load(manifest_path)} if manifest_path.exists() else {}
    merged.update({row["index"]: row for row in manifest})
    with manifest_path.open("w") as handle:
        for index in sorted(merged):
            handle.write(json.dumps(merged[index]) + "\n")
    print(f"wrote {len(manifest)} packets to {out}, {missing} without a located file")
