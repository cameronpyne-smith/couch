import json
import random
import re
import sys
from collections import Counter

from transformers import AutoTokenizer

from projects.pr_reviews.contained import CONTAINED, contained, key, label, load
from projects.pr_reviews.dataset import EXPORT, load_records, locate, render, split_examples, write_jsonl
from projects.pr_reviews.generator import NO_BREAK, OUTPUT, frame

CONTEXT = OUTPUT.with_name(OUTPUT.name + "-context")
SUBSET = CONTEXT.with_name(CONTEXT.name + "-contained")
TOKENISER = "Qwen/Qwen3.8-27B"
MAX_DIFF_CHARS = 7000
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
PROMPT = (
    "You are reviewing one function from a pull request. The function is shown first, then the pull request's changes to its file "
    "as a unified diff. Find one concrete sequence of events under which the function breaks what it promises its callers and the "
    "rest of the system, and quote the code where it goes wrong. If there is none, answer: no break found."
)
RULE = (
    "You are shown one function, the pull request's changes to its file as a unified diff, and a code review finding. Decide whether "
    "a reviewer who sees only this text, and nothing else from the repository, could verify the finding. It is contained when the "
    "problem it describes is visible in the code or diff shown. It is not contained when checking it needs other files, callers, "
    "tests, or declarations defined elsewhere. "
    'Reply with JSON only: {"contained": true} or {"contained": false}.'
)


def hunks_for(diff_path, path):
    sections = ("\n" + diff_path.read_text(errors="replace")).split("\ndiff --git ")[1:]
    for section in sections:
        header, *hunks = section.split("\n@@ ")
        if header.split("\n", 1)[0].endswith(f" b/{path}"):
            return ["@@ " + hunk.rstrip("\n") for hunk in hunks]
    return []


def distance(hunk, line):
    match = HUNK.match(hunk)
    start = int(match.group(1))
    end = start + max(int(match.group(2) or 1), 1) - 1
    return 0 if start <= line <= end else min(abs(line - start), abs(line - end))


def nearest(hunks, line, budget=MAX_DIFF_CHARS):
    kept = set()
    used = 0
    for index in sorted(range(len(hunks)), key=lambda i: distance(hunks[i], line)):
        if used + len(hunks[index]) > budget and kept:
            break
        kept.add(index)
        used += len(hunks[index])
    return "\n".join(hunks[i][:budget] for i in sorted(kept))


def with_diff(unit_prompt, diff):
    return f"{unit_prompt}\n\n### Changes to this file in the pull request\n```diff\n{diff}\n```"


def build_real(record):
    finding = record["finding"]
    hunks = hunks_for(record["diff"], finding["path"])
    if not hunks:
        return None
    unit = render(record["text"], record["span"], numbered=False)
    numbered = re.search(r"\blines? \d+", finding["body"], re.IGNORECASE) is not None
    return {
        "system": PROMPT,
        "prompt": with_diff(frame(unit, record["span"][0], finding["path"], finding["id"], numbered), nearest(hunks, finding["line"])),
        "answer": finding["body"],
        "kind": "real",
        "labelled": finding["labelled"],
        "repo": record["repo"],
        "number": record["number"],
        "reviewed_at": record["reviewed_at"],
    }


def build_clean(function, diff_paths):
    diff_path = diff_paths[(function["repo"], function["number"], function["commit"])]
    hunks = hunks_for(diff_path, function["path"])
    if not hunks:
        return None
    seed = f"{function['repo']}#{function['number']}:{function['path']}:{function['span'][0]}"
    return {
        "system": PROMPT,
        "prompt": with_diff(frame(function["unit"], function["span"][0], function["path"], seed), nearest(hunks, function["span"][0] + 1)),
        "answer": NO_BREAK,
        "kind": "clean",
        "labelled": "copilot:silent",
        "repo": function["repo"],
        "number": function["number"],
        "reviewed_at": function["reviewed_at"],
    }


def clean_examples(pool):
    diff_paths = {}
    for function in pool:
        ident = (function["repo"], function["number"], function["commit"])
        if ident not in diff_paths:
            pr_dir = EXPORT / function["repo"] / str(function["number"])
            pr = json.loads((pr_dir / "pr.json").read_text())
            diff_paths[ident] = pr_dir / pr["diffs"][function["commit"]]
    return [example for function in pool if (example := build_clean(function, diff_paths)) is not None]


def ask(prompt, finding):
    return contained(prompt, finding, rule=RULE, prompt_chars=14000, num_ctx=8192)


def lengths(name, examples, tokeniser):
    tokens = sorted(len(tokeniser(e["system"] + e["prompt"]).input_ids) for e in examples)
    print(f"{name}: {len(examples)} prompt tokens median {tokens[len(tokens) // 2]} p90 {tokens[int(len(tokens) * 0.9)]} max {tokens[-1]}", flush=True)


def subset():
    SUBSET.mkdir(exist_ok=True)
    verdicts = {row["key"]: row["contained"] for row in load(CONTEXT / "verdicts.jsonl")}
    tokeniser = AutoTokenizer.from_pretrained(TOKENISER)
    for name in ("train", "val"):
        examples = load(CONTEXT / f"{name}.jsonl")
        real = [e for e in examples if e["kind"] == "real" and verdicts[key(e)]]
        clean = random.Random(0).sample([e for e in examples if e["kind"] == "clean"], len(real))
        kept = [e for e in examples if e in real or e in clean]
        write_jsonl(SUBSET / f"{name}.jsonl", kept)
        print(f"{name}: real {len(real)}/{sum(e['kind'] == 'real' for e in examples)} clean {len(clean)} kept {len(kept)}", flush=True)
        lengths(name, kept, tokeniser)


if __name__ == "__main__" and sys.argv[1:] == ["subset"]:
    subset()
elif __name__ == "__main__":
    CONTEXT.mkdir(exist_ok=True)
    records = [r for r in load_records() if r["finding"]["verdict"] == "valid"]
    located = [r for r in records if locate(r) == "function" and r["diff"] is not None]
    real = [example for r in located if (example := build_real(r)) is not None]
    print("valid findings", len(records), "with function", len(located), "with file diff", len(real), flush=True)

    pool = load(OUTPUT / "pool.jsonl")
    clean = clean_examples(pool)
    print("clean pool", len(pool), "with file diff", len(clean), flush=True)
    clean = random.Random(0).sample(clean, len(real))

    tokeniser = AutoTokenizer.from_pretrained(TOKENISER)
    lengths("real", real, tokeniser)
    lengths("clean", clean, tokeniser)

    splits = split_examples(real + clean)
    for name, split in splits.items():
        print(name, len(split), "by kind", dict(Counter(e["kind"] for e in split)), flush=True)
        write_jsonl(CONTEXT / f"{name}.jsonl", split)

    sample = random.Random(0).sample([e for e in splits["train"] if e["kind"] == "real"], 2)
    for e in sample:
        print("=" * 80)
        print(e["prompt"][:3000])
        print("-> " + e["answer"][:400])

    verdicts = label({name: splits[name] for name in ("train", "val")}, CONTEXT / "verdicts.jsonl", ask)
    old = {row["key"]: row["contained"] for row in load(CONTAINED / "verdicts.jsonl")}
    for name in ("train", "val"):
        reals = [e for e in splits[name] if e["kind"] == "real"]
        new_share = sum(verdicts[key(e)] for e in reals)
        both = [e for e in reals if key(e) in old]
        old_share = sum(old[key(e)] for e in both)
        print(f"{name}: contained with diff {new_share}/{len(reals)} ({new_share / len(reals):.0%}); function only {old_share}/{len(both)} ({old_share / len(both):.0%}) on the {len(both)} findings present in both", flush=True)
        print("   ", dict(Counter((old[key(e)], verdicts[key(e)]) for e in both)), "as (function only, with diff)", flush=True)

    reals = [e for s in ("train", "val") for e in splits[s] if e["kind"] == "real" and key(e) in old]
    for title, wanted in (("newly contained", (False, True)), ("still not contained", (False, False))):
        print("=" * 80, title)
        for e in random.Random(0).sample([e for e in reals if (old[key(e)], verdicts[key(e)]) == wanted], 3):
            print("-> " + e["answer"][:300])
