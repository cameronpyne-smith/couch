import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

from finetune.dataset import (
    DECLARATION_WRAPPERS,
    EXPORT,
    GOLDEN,
    LANGUAGES,
    LINE_REFERENCE,
    MAX_TYPE_LINES,
    PARSERS,
    git_show,
    is_function,
    load_records,
    locate,
    render,
    split_examples,
    write_jsonl,
)

OUTPUT = Path.home() / "training-data" / "code" / "pr-reviews-generator"
NO_BREAK = "no break found"
UNIT_PROMPT = (
    "You are reviewing one function. Find one concrete sequence of events under which it breaks what it promises "
    "its callers and the rest of the system, and quote the code where it goes wrong. "
    "If there is none, answer: no break found."
)
OUTSIDE_REFERENCE = re.compile(r"#\d+|\bPR\b|pull request|\.(cs|ts|svelte|json|ya?ml)\b|commit", re.IGNORECASE)
DIFF_HEADER = re.compile(r"^diff --git a/(.*) b/(.*)$", re.MULTILINE)
EXCLUDED_PATHS = re.compile(r"test|example|docs?/", re.IGNORECASE)
MIN_FUNCTION_LINES = 4
MAX_PER_FILE = 8


def frame(unit, start, path, seed, numbered=False):
    style = random.Random(seed)
    numbered = style.random() < 0.5 or numbered
    if numbered:
        unit = "\n".join(f"{start + i + 1:5d}   {line}" for i, line in enumerate(unit.split("\n")))
    header = f"### {path}\n" if style.random() < 0.5 else ""
    return header + unit


def build_real(record):
    finding = record["finding"]
    unit = render(record["text"], record["span"], numbered=False)
    numbered = LINE_REFERENCE.search(finding["body"]) is not None
    return {
        "system": UNIT_PROMPT,
        "prompt": frame(unit, record["span"][0], finding["path"], finding["id"], numbered),
        "answer": finding["body"],
        "kind": "real",
        "labelled": finding["labelled"],
        "repo": record["repo"],
        "number": record["number"],
        "reviewed_at": record["reviewed_at"],
    }


def build_clean(function):
    seed = f"{function['repo']}#{function['number']}:{function['path']}:{function['span'][0]}"
    return {
        "system": UNIT_PROMPT,
        "prompt": frame(function["unit"], function["span"][0], function["path"], seed),
        "answer": NO_BREAK,
        "kind": "clean",
        "labelled": "copilot:silent",
        "repo": function["repo"],
        "number": function["number"],
        "reviewed_at": function["reviewed_at"],
    }


def sample_clean(pool, count, seed=0):
    by_pr = {}
    for function in pool:
        by_pr.setdefault((function["repo"], function["number"]), []).append(function)
    groups = list(by_pr.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(group)
    picked = []
    depth = 0
    while len(picked) < count:
        for group in groups:
            if depth < len(group):
                picked.append(group[depth])
        depth += 1
    return [build_clean(function) for function in picked[:count]]


def load_prs():
    prs = []
    for golden_path in GOLDEN.rglob("*.json"):
        golden = json.loads(golden_path.read_text())
        pr_dir = EXPORT / golden["repo"] / str(golden["number"])
        pr = json.loads((pr_dir / "pr.json").read_text())
        if golden["commit"] not in pr["diffs"]:
            continue
        diff = pr_dir / pr["diffs"][golden["commit"]]
        paths = sorted({new for _, new in DIFF_HEADER.findall(diff.read_text(errors="replace"))})
        prs.append(
            {
                "repo": golden["repo"],
                "number": golden["number"],
                "reviewed_at": golden["reviewed_at"],
                "commit": golden["commit"],
                "paths": paths,
                "finding_paths": {f["path"] for f in golden.get("findings") or []},
            }
        )
    return prs


def function_spans(root, language, offset=0):
    spans = []
    stack = [root]
    while stack:
        node = stack.pop()
        if is_function(node, language):
            while node.parent is not None and node.parent.type in DECLARATION_WRAPPERS:
                node = node.parent
            spans.append((node.start_point.row + offset, node.end_point.row + offset))
            continue
        stack.extend(reversed(node.children))
    return spans


def functions_in(text, language):
    source = text.encode()
    root = PARSERS[language].parse(source).root_node
    if language != "svelte":
        return function_spans(root, language)
    spans = []
    for element in root.children:
        if element.type != "script_element":
            continue
        script = next(child for child in element.children if child.type == "raw_text")
        inner = PARSERS["ts"].parse(source[script.start_byte : script.end_byte]).root_node
        spans.extend(function_spans(inner, "ts", script.start_point.row))
    return spans


def clean_functions(prs):
    pool = []
    seen = set()
    for pr in prs:
        for path in pr["paths"]:
            language = LANGUAGES.get(Path(path).suffix)
            if language is None or EXCLUDED_PATHS.search(path) or path in pr["finding_paths"]:
                continue
            try:
                text = git_show(pr["repo"], pr["commit"], path)
            except subprocess.CalledProcessError:
                continue
            spans = [
                span
                for span in functions_in(text, language)
                if MIN_FUNCTION_LINES <= span[1] - span[0] + 1 < MAX_TYPE_LINES
            ]
            picked = random.Random(f"{pr['repo']}#{pr['number']}:{path}").sample(spans, min(len(spans), MAX_PER_FILE))
            for span in sorted(picked):
                unit = render(text, span, numbered=False)
                if unit in seen:
                    continue
                seen.add(unit)
                pool.append(
                    {
                        "repo": pr["repo"],
                        "number": pr["number"],
                        "reviewed_at": pr["reviewed_at"],
                        "commit": pr["commit"],
                        "path": path,
                        "language": language,
                        "span": span,
                        "unit": unit,
                    }
                )
    return pool


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

    prs = load_prs()
    pool = clean_functions(prs)
    print("PRs", len(prs), "clean functions", len(pool), "from", len({(p["repo"], p["number"]) for p in pool}), "PRs")
    print("by language", dict(Counter(p["language"] for p in pool)))
    lines = sorted(p["span"][1] - p["span"][0] + 1 for p in pool)
    print("lines median", lines[len(lines) // 2], "p90", lines[int(len(lines) * 0.9)])
    for p in random.Random(0).sample(pool, 5):
        print("=" * 80)
        print(p["path"], p["span"])
        print(p["unit"])
    write_jsonl(OUTPUT / "pool.jsonl", pool)
    print("wrote", len(pool), "to", OUTPUT / "pool.jsonl")

    clean = sample_clean(pool, len(examples))
    print("clean", len(clean), "from", len({(e["repo"], e["number"]) for e in clean}), "PRs")
    for e in random.Random(0).sample(clean, 3):
        print("=" * 80)
        print(e["prompt"])
        print("-> " + e["answer"])
    write_jsonl(OUTPUT / "clean.jsonl", clean)
    print("wrote", len(clean), "to", OUTPUT / "clean.jsonl")

    for name, split in split_examples(examples + clean).items():
        prs = {(e["repo"], e["number"]) for e in split}
        dates = sorted(e["reviewed_at"][:10] for e in split)
        print(name, len(split), "from", len(prs), "PRs", dates[0], "to", dates[-1], "by kind", dict(Counter(e["kind"] for e in split)))
        write_jsonl(OUTPUT / f"{name}.jsonl", split)
