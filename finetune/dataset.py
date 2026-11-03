import json
import subprocess
from collections import Counter
from pathlib import Path

import tree_sitter_c_sharp
from tree_sitter import Language, Parser


GOLDEN = Path.home() / "code" / "reviewdo" / "bench" / "golden"
EXPORT = Path.home() / "training-data" / "code" / "pr-reviews"
CLONES = Path.home() / "code" / "remundo"
VERDICTS = ("valid", "invalid")
PARSERS = {".cs": Parser(Language(tree_sitter_c_sharp.language()))}
FUNCTION_NODES = {
    ".cs": ("method_declaration", "constructor_declaration", "local_function_statement", "property_declaration"),
}
TYPE_NODES = {
    ".cs": ("class_declaration", "record_declaration", "struct_declaration", "interface_declaration", "enum_declaration", "compilation_unit"),
}
MAX_TYPE_LINES = 120


def load_records():
    records = []
    for golden_path in GOLDEN.rglob("*.json"):
        golden = json.loads(golden_path.read_text())
        pr_dir = EXPORT / golden["repo"] / str(golden["number"])
        pr = json.loads((pr_dir / "pr.json").read_text())
        for finding in golden.get("findings") or []:
            if finding["verdict"] not in VERDICTS or "duplicate_of" in finding:
                continue
            diff_name = pr["diffs"].get(finding["commit"])
            records.append(
                {
                    "repo": golden["repo"],
                    "number": golden["number"],
                    "title": golden["title"],
                    "author": golden["author"],
                    "base_ref": golden["base_ref"],
                    "reviewed_at": golden["reviewed_at"],
                    "description": pr["body"],
                    "finding": finding,
                    "diff": pr_dir / diff_name if diff_name else None,
                }
            )
    return records


def read_file_at_commit(record):
    clone = CLONES / record["repo"].split("/", 1)[1]
    finding = record["finding"]
    result = subprocess.run(
        ["git", "-C", str(clone), "show", f"{finding['commit']}:{finding['path']}"],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    return result.stdout


def enclosing_function(text, line, suffix):
    tree = PARSERS[suffix].parse(text.encode())
    row = line - 1
    node = tree.root_node.descendant_for_point_range((row, 0), (row, 0))
    fallback = None
    while node is not None and node.type not in FUNCTION_NODES[suffix]:
        if fallback is None and node.type in TYPE_NODES[suffix]:
            fallback = node
        node = node.parent
    if node is None:
        node = fallback
    if node is None or node.end_point.row - node.start_point.row >= MAX_TYPE_LINES:
        return None
    return node.start_point.row, node.end_point.row


def locate(record):
    suffix = Path(record["finding"]["path"]).suffix
    if suffix not in PARSERS:
        return "no language"
    try:
        text = read_file_at_commit(record)
    except subprocess.CalledProcessError:
        return "no file"
    span = enclosing_function(text, record["finding"]["line"], suffix)
    if span is None:
        return "no function"
    record["text"] = text
    record["span"] = span
    return "function"


def render(text, span, numbered=True):
    lines = text.splitlines()
    start, end = span
    if numbered:
        return "\n".join(f"{n + 1:5d}   {lines[n]}" for n in range(start, end + 1))
    return "\n".join(lines[start : end + 1])


if __name__ == "__main__":
    records = load_records()
    print("by verdict", Counter(r["finding"]["verdict"] for r in records))
    print("by label source", Counter(r["finding"]["labelled"].split(":")[0] for r in records))
    print("without diff", sum(r["diff"] is None for r in records))

    print("located", Counter(locate(r) for r in records))

    example = next(r for r in records if r["repo"].endswith("Pricing.Employee") and r["number"] == 13)
    print(render(example["text"], example["span"]))
