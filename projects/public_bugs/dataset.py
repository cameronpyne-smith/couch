import argparse
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

from projects.pr_reviews.context import HUNK, nearest, with_diff
from projects.pr_reviews.dataset import LANGUAGES, enclosing_function, render, write_jsonl
from projects.pr_reviews.generator import MIN_FUNCTION_LINES, frame
from projects.pr_reviews.traces import NO_BREAK, SYSTEM

PUBLIC = Path.home() / "training-data" / "code" / "public-bugs"
CLONES = PUBLIC / "clones"
DATA = PUBLIC / "dataset"
MAX_PROMPT_CHARS = 16_000
MIN_SHARE = 0.5
MAX_INTRODUCED_FILES = 10
MAX_ISSUE_CHARS = 1500
MAX_FIX_DIFF_CHARS = 4000
PORCELAIN_HEADER = re.compile(r"^([0-9a-f]{40}) (\d+) (\d+)")


def clone_of(repo):
    return CLONES / repo.replace("/", "__")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(clone_of(repo)), *args], capture_output=True, text=True, errors="replace", check=True).stdout


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def blame(repo, commit, path, ranges):
    lines = []
    for start, end in ranges:
        try:
            out = git(repo, "blame", "-w", "--line-porcelain", "-L", f"{start},{end}", commit, "--", path)
        except subprocess.CalledProcessError:
            continue
        sha = filename = original = None
        for line in out.splitlines():
            header = PORCELAIN_HEADER.match(line)
            if header:
                sha, original = header.group(1), int(header.group(2))
            elif line.startswith("filename "):
                filename = line[len("filename ") :]
            elif line.startswith("\t"):
                lines.append((sha, filename, original))
    return lines


def hunks_of(repo, commit, path):
    try:
        diff = git(repo, "diff", f"{commit}^", commit, "--", path)
    except subprocess.CalledProcessError:
        return []
    return ["@@ " + hunk.rstrip("\n") for hunk in diff.split("\n@@ ")[1:]]


def show(repo, commit, path):
    try:
        return git(repo, "show", f"{commit}:{path}")
    except subprocess.CalledProcessError:
        return None


def locate(repo, commit, path, lines):
    language = LANGUAGES.get(Path(path).suffix)
    text = show(repo, commit, path) if language else None
    if text is None:
        return None
    spans = Counter()
    for line in lines:
        span = enclosing_function(text, line, language)
        if span is not None and span[1] - span[0] + 1 >= MIN_FUNCTION_LINES:
            spans[span] += 1
    if not spans:
        return None
    span = spans.most_common(1)[0][0]
    return text, span, min(line for line in lines if enclosing_function(text, line, language) == span)


def gold_for(fix):
    parts = [f"Fixed later by pull request #{fix['pr']}: {fix['title']}"]
    if fix.get("issue"):
        issue = fix["issue"]
        parts.append(f"Issue #{issue['number']}: {issue['title']}\n{(issue.get('body') or '')[:MAX_ISSUE_CHARS]}".rstrip())
    parts.append(f"Fix diff:\n```diff\n{fix['diff'][:MAX_FIX_DIFF_CHARS]}\n```")
    return "\n\n".join(parts)


def build_real(fix):
    if not fix.get("introduced_commit") or fix["introduced_commit"] == fix["fix_commit"]:
        return None, "no introducing commit"
    if (fix.get("introduced_share") or 0) < MIN_SHARE or (fix.get("introduced_files") or 0) > MAX_INTRODUCED_FILES:
        return None, "weak introducing commit"
    by_file = {}
    for entry in fix["files"]:
        for sha, filename, original in blame(fix["repo"], f"{fix['fix_commit']}~1", entry["path"], entry.get("old_ranges") or []):
            if sha == fix["introduced_commit"]:
                by_file.setdefault(filename, []).append(original)
    if not by_file:
        return None, "blame disagrees"
    path, lines = max(by_file.items(), key=lambda item: len(item[1]))
    located = locate(fix["repo"], fix["introduced_commit"], path, lines)
    if located is None:
        return None, "no function"
    text, span, line = located
    hunks = hunks_of(fix["repo"], fix["introduced_commit"], path)
    if not hunks:
        return None, "no diff"
    unit = render(text, span, numbered=False)
    prompt = with_diff(frame(unit, span[0], path, f"{fix['repo']}#{fix['pr']}"), nearest(hunks, line))
    if len(prompt) > MAX_PROMPT_CHARS:
        return None, "too long"
    return {
        "system": SYSTEM,
        "prompt": prompt,
        "answer": gold_for(fix),
        "kind": "real",
        "repo": fix["repo"],
        "number": fix.get("introduced_pr"),
        "commit": fix["introduced_commit"],
        "path": path,
        "line": line,
        "fix_pr": fix["pr"],
        "fix_commit": fix["fix_commit"],
        "merged_at": fix["merged_at"],
    }, "built"


def changed_lines(hunks):
    lines = []
    for hunk in hunks:
        match = HUNK.match(hunk)
        new = int(match.group(1))
        for line in hunk.split("\n")[1:]:
            if line.startswith("+"):
                lines.append(new)
            if not line.startswith("-"):
                new += 1
    return lines


def build_clean(merge, path, rng):
    hunks = hunks_of(merge["repo"], merge["merge_commit"], path)
    lines = changed_lines(hunks)
    if not lines:
        return None
    located = locate(merge["repo"], merge["merge_commit"], path, rng.sample(lines, min(len(lines), 12)))
    if located is None:
        return None
    text, span, line = located
    unit = render(text, span, numbered=False)
    prompt = with_diff(frame(unit, span[0], path, f"{merge['repo']}#{merge['pr']}:{path}"), nearest(hunks, line))
    if len(prompt) > MAX_PROMPT_CHARS:
        return None
    return {
        "system": SYSTEM,
        "prompt": prompt,
        "answer": NO_BREAK,
        "kind": "clean",
        "repo": merge["repo"],
        "number": merge["pr"],
        "commit": merge["merge_commit"],
        "path": path,
        "line": line,
        "merged_at": merge["merged_at"],
    }


def later_fixed(fixes):
    touched = {}
    for fix in fixes:
        for entry in fix["files"]:
            touched.setdefault((fix["repo"], entry["path"]), []).append(fix["merged_at"])
    return touched


def key_of(example):
    return (example["repo"], example["commit"], example["path"], example["line"])


def build(val_repos, clean_share, seed, per_repo, extend):
    fixes = load(PUBLIC / "fixes.jsonl")
    merges = load(PUBLIC / "merges.jsonl")
    licences = {row["repo"]: row["licence"] for row in load(PUBLIC / "repos.jsonl")}
    existing = load(DATA / "train.jsonl") + load(DATA / "val.jsonl") if extend else []
    seen = {key_of(e) for e in existing}
    have = Counter((e["repo"], e["kind"]) for e in existing)
    rng = random.Random(seed)
    reasons = Counter()
    reals = []
    by_repo = {}
    for fix in fixes:
        if not extend or fix["repo"] not in val_repos:
            by_repo.setdefault(fix["repo"], []).append(fix)
    for repo, candidates in sorted(by_repo.items()):
        rng.shuffle(candidates)
        built = have[(repo, "real")]
        for fix in candidates:
            if built >= per_repo:
                break
            example, reason = build_real(fix)
            reasons[reason] += 1
            if example and key_of(example) not in seen:
                seen.add(key_of(example))
                reals.append(example)
                built += 1
        print(f"  {repo}: {built} real", flush=True)
    print("real", dict(reasons))
    touched = later_fixed(fixes)
    candidates = []
    for merge in merges:
        for path in merge["files"]:
            if Path(path).suffix in LANGUAGES and not any(when > merge["merged_at"] for when in touched.get((merge["repo"], path), [])) and not (extend and merge["repo"] in val_repos):
                candidates.append((merge, path))
    rng.shuffle(candidates)
    total_real = len(reals) + sum(n for (_, kind), n in have.items() if kind == "real")
    wanted = round(total_real * clean_share / (1 - clean_share)) - sum(n for (_, kind), n in have.items() if kind == "clean")
    cleans = []
    for merge, path in candidates:
        if len(cleans) >= wanted:
            break
        example = build_clean(merge, path, rng)
        if example and key_of(example) not in seen:
            seen.add(key_of(example))
            cleans.append(example)
    print("clean", len(cleans), "of", len(candidates), "candidates, wanted", wanted)
    DATA.mkdir(parents=True, exist_ok=True)
    for split, keep in (("train", lambda e: e["repo"] not in val_repos), ("val", lambda e: e["repo"] in val_repos)):
        rows = [{**e, "licence": licences.get(e["repo"])} for e in reals + cleans if keep(e)]
        rng.shuffle(rows)
        if extend:
            if split == "val":
                continue
            rows = load(DATA / "train.jsonl") + rows
        write_jsonl(DATA / f"{split}.jsonl", rows)
        print(split, dict(Counter(e["kind"] for e in rows)), dict(Counter(e["repo"] for e in rows)))


def stats():
    fixes = load(PUBLIC / "fixes.jsonl")
    print("fixes", len(fixes), "with introducing commit", sum(1 for f in fixes if f.get("introduced_commit")))
    for repo, count in sorted(Counter(f["repo"] for f in fixes).items()):
        strong = sum(1 for f in fixes if f["repo"] == repo and f.get("introduced_commit") and (f.get("introduced_share") or 0) >= MIN_SHARE and (f.get("introduced_files") or 0) <= MAX_INTRODUCED_FILES)
        print(f"  {repo:40s} {count:4d} fixes, {strong:4d} strong")
    if (PUBLIC / "merges.jsonl").exists():
        print("merges", dict(Counter(m["repo"] for m in load(PUBLIC / "merges.jsonl"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("stats", "build"))
    parser.add_argument("--val-repos", nargs="*", default=[])
    parser.add_argument("--clean-share", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-repo", type=int, default=60)
    parser.add_argument("--extend", action="store_true", help="keep the existing rows and append new ones to train only")
    args = parser.parse_args()
    if args.command == "stats":
        stats()
    else:
        build(set(args.val_repos), args.clean_share, args.seed, args.per_repo, args.extend)
