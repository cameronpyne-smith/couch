import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

import tree_sitter_c_sharp
import tree_sitter_svelte
import tree_sitter_typescript
from tree_sitter import Language, Parser


GOLDEN = Path.home() / "code" / "reviewdo" / "bench" / "golden"
EXPORT = Path.home() / "training-data" / "code" / "pr-reviews"
CLONES = Path.home() / "code" / "remundo"
VERDICTS = ("valid", "invalid")
LANGUAGES = {".cs": "cs", ".ts": "ts", ".js": "ts", ".mjs": "ts", ".svelte": "svelte"}
PARSERS = {
    "cs": Parser(Language(tree_sitter_c_sharp.language())),
    "ts": Parser(Language(tree_sitter_typescript.language_typescript())),
    "svelte": Parser(Language(tree_sitter_svelte.language())),
}
FUNCTION_NODES = {
    "cs": ("method_declaration", "constructor_declaration", "local_function_statement", "property_declaration"),
    "ts": ("function_declaration", "method_definition", "arrow_function", "function_expression"),
}
ASSIGNED_ONLY = ("arrow_function", "function_expression")
DECLARATION_WRAPPERS = ("variable_declarator", "lexical_declaration", "export_statement")
TYPE_NODES = {
    "cs": ("class_declaration", "record_declaration", "struct_declaration", "interface_declaration", "enum_declaration", "compilation_unit"),
    "ts": ("class_declaration", "abstract_class_declaration", "interface_declaration", "program"),
}
MAX_TYPE_LINES = 120
MAX_PROMPT_CHARS = 16_000
LINE_REFERENCE = re.compile(r"\blines? \d+", re.IGNORECASE)
SYSTEM = (
    "You are reviewing one function from a pull request. A reviewer has raised a finding against it. "
    "Decide whether the finding is valid: a real problem in this code that the author should fix. "
    "Answer with one word, valid or invalid."
)


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


def is_function(node, language):
    if node.type not in FUNCTION_NODES[language]:
        return False
    if node.type in ASSIGNED_ONLY:
        return node.parent.type == "variable_declarator"
    return True


def enclosing_in_tree(root, row, language):
    node = root.descendant_for_point_range((row, 0), (row, 0))
    fallback = None
    while node is not None and not is_function(node, language):
        if fallback is None and node.type in TYPE_NODES[language]:
            fallback = node
        node = node.parent
    if node is None:
        node = fallback
    if node is None or node.end_point.row - node.start_point.row >= MAX_TYPE_LINES:
        return None
    while node.parent is not None and node.parent.type in DECLARATION_WRAPPERS:
        node = node.parent
    return node.start_point.row, node.end_point.row


def enclosing_in_svelte(source, root, row):
    for element in root.children:
        if element.type != "script_element":
            continue
        script = next(child for child in element.children if child.type == "raw_text")
        if not script.start_point.row <= row <= script.end_point.row:
            continue
        offset = script.start_point.row
        inner = PARSERS["ts"].parse(source[script.start_byte : script.end_byte])
        span = enclosing_in_tree(inner.root_node, row - offset, "ts")
        if span is not None:
            return span[0] + offset, span[1] + offset
    if root.end_point.row - root.start_point.row >= MAX_TYPE_LINES:
        return None
    return root.start_point.row, root.end_point.row


def enclosing_function(text, line, language):
    source = text.encode()
    root = PARSERS[language].parse(source).root_node
    row = line - 1
    if language == "svelte":
        return enclosing_in_svelte(source, root, row)
    return enclosing_in_tree(root, row, language)


def locate(record):
    language = LANGUAGES.get(Path(record["finding"]["path"]).suffix)
    if language is None:
        return "no language"
    try:
        text = read_file_at_commit(record)
    except subprocess.CalledProcessError:
        return "no file"
    span = enclosing_function(text, record["finding"]["line"], language)
    if span is None:
        return "no function"
    record["text"] = text
    record["span"] = span
    return "function"


def render(text, span, numbered=True):
    lines = text.split("\n")
    start, end = span
    if numbered:
        return "\n".join(f"{n + 1:5d}   {lines[n]}" for n in range(start, end + 1))
    return "\n".join(lines[start : end + 1])


def build_example(record, max_prompt_chars=MAX_PROMPT_CHARS):
    finding = record["finding"]
    style = random.Random(finding["id"])
    numbered = style.random() < 0.5 or LINE_REFERENCE.search(finding["body"]) is not None
    unit = render(record["text"], record["span"], numbered=numbered)
    header = f"### {finding['path']}\n" if style.random() < 0.5 else ""
    prompt = f"{header}{unit}\n\nFinding:\n{finding['body']}"
    if len(prompt) > max_prompt_chars:
        return None
    return {
        "system": SYSTEM,
        "prompt": prompt,
        "answer": finding["verdict"],
        "labelled": finding["labelled"],
        "repo": record["repo"],
        "number": record["number"],
        "reviewed_at": record["reviewed_at"],
    }


if __name__ == "__main__":
    records = load_records()
    print("by verdict", Counter(r["finding"]["verdict"] for r in records))
    print("by label source", Counter(r["finding"]["labelled"].split(":")[0] for r in records))
    print("without diff", sum(r["diff"] is None for r in records))

    print("located", Counter(locate(r) for r in records))
    print("by language", Counter((LANGUAGES.get(Path(r["finding"]["path"]).suffix), "span" in r) for r in records))

    examples = [build_example(r) for r in records if "span" in r]
    print("too long", sum(e is None for e in examples))
    examples = [e for e in examples if e is not None]
    for e in random.Random(0).sample(examples, 3):
        print("=" * 80)
        print(e["prompt"])
        print("-> " + e["answer"], e["labelled"], e["repo"], e["number"])
