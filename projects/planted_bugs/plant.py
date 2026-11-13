import argparse
import collections
import json
import random
import re
import subprocess
from pathlib import Path

PUBLIC = Path.home() / "training-data" / "code" / "public-bugs"
CLONES = PUBLIC / "clones"
OUT = Path(__file__).parent / "data" / "planted-bugs.jsonl"
VAL_REPOS = ["colinhacks/zod", "Humanizr/Humanizer", "quartznet/quartznet"]
MIN_LINES = 6
MAX_LINES = 60

MUTATIONS = [
    ("lt-le", re.compile(r"(?<=\s)<(?=\s)"), "<="),
    ("le-lt", re.compile(r"(?<=\s)<=(?=\s)"), "<"),
    ("gt-ge", re.compile(r"(?<=\s)>(?=\s)"), ">="),
    ("ge-gt", re.compile(r"(?<=\s)>=(?=\s)"), ">"),
    ("eq-ne", re.compile(r"(?<=\s)(===?)(?=\s)"), lambda m: "!" + m.group(1)[1:]),
    ("ne-eq", re.compile(r"(?<=\s)!(==?)(?=\s)"), lambda m: "=" + m.group(1)),
    ("and-or", re.compile(r"(?<=\s)&&(?=\s)"), "||"),
    ("or-and", re.compile(r"(?<=\s)\|\|(?=\s)"), "&&"),
    ("drop-not", re.compile(r"(?<=[(\s])!(?=[A-Za-z_(])"), ""),
    ("drop-plus-one", re.compile(r" \+ 1(?![0-9.])"), ""),
    ("drop-minus-one", re.compile(r" - 1(?![0-9.])"), ""),
    ("true-false", re.compile(r"\btrue\b"), "false"),
    ("false-true", re.compile(r"\bfalse\b"), "true"),
    ("null-coalesce", re.compile(r"\?\.(?=[A-Za-z_])"), "."),
]

KINDS = {".cs": ("cs", {"method_declaration", "constructor_declaration", "local_function_statement"}), ".ts": ("ts", {"function_declaration", "method_definition"})}
SKIP = re.compile(r"(^|/)(test|tests|spec|__tests__|benchmarks?|samples?|examples?|docs?)(/|$)|\.(test|spec|d)\.ts$|Tests?\.cs$|\.g\.cs$|Generated", re.I)


def code_end(line):
    cut = len(line)
    for marker in ("//", "/*", " * "):
        at = line.find(marker)
        if at >= 0:
            cut = min(cut, at)
    return cut


def outside_strings(line, at):
    before = line[:at]
    return all(before.count(quote) % 2 == 0 for quote in ('"', "'", "`"))


def sites(function):
    out = []
    for i, (_, text) in enumerate(function):
        stripped = text.lstrip()
        if not stripped or stripped.startswith(("//", "*", "/*", "using ", "import ", "#")):
            continue
        end = code_end(text)
        for name, pattern, replacement in MUTATIONS:
            for match in pattern.finditer(text):
                if match.start() < end and outside_strings(text, match.start()):
                    out.append((i, name, match, replacement))
    return out


def plant(function, site):
    i, name, match, replacement = site
    text = function[i][1]
    new = replacement(match) if callable(replacement) else replacement
    planted = list(function)
    planted[i] = (function[i][0], text[: match.start()] + new + text[match.end() :])
    return planted


def render(function):
    width = len(str(function[-1][0]))
    return "\n".join(f"{number:>{width}}  {text}" for number, text in function)


def parsers():
    import tree_sitter_c_sharp
    import tree_sitter_typescript
    from tree_sitter import Language, Parser

    return {"cs": Parser(Language(tree_sitter_c_sharp.language())), "ts": Parser(Language(tree_sitter_typescript.language_typescript()))}


def functions_of(repo, parse, cap, rng):
    root = CLONES / repo.replace("/", "__")
    files = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    files = [f for f in files if Path(f).suffix in KINDS and not SKIP.search(f)]
    rng.shuffle(files)
    out = []
    for f in files:
        language, wanted = KINDS[Path(f).suffix]
        if not (root / f).is_file() or (root / f).stat().st_size > 400_000:
            continue
        source = (root / f).read_bytes()
        if not source:
            continue
        stack = [parse[language].parse(source).root_node]
        while stack:
            node = stack.pop()
            if node.type in wanted:
                text = source[node.start_byte : node.end_byte].decode("utf-8", "replace")
                indent = source[: node.start_byte].rsplit(b"\n", 1)[-1].decode("utf-8", "replace")
                lines = (indent + text).splitlines()
                if MIN_LINES <= len(lines) <= MAX_LINES:
                    out.append((f, [(k + 1, line.rstrip()) for k, line in enumerate(lines)]))
            stack.extend(node.children)
        if len(out) >= cap * 3:
            break
    return out


def build(split, repos, licences, cap, rng, parse):
    rows, kinds = [], collections.Counter()
    for repo in repos:
        seen, taken = set(), 0
        for path, function in functions_of(repo, parse, cap, rng):
            key = "\n".join(text.strip() for _, text in function)
            if key in seen or taken >= cap:
                continue
            seen.add(key)
            candidates = sites(function)
            if not candidates:
                continue
            site = rng.choice(candidates)
            planted = plant(function, site)
            i, name = site[0], site[1]
            pair = len(rows) // 2
            base = {"split": split, "pair": pair, "repo": repo, "path": path, "licence": licences[repo]}
            rows.append({**base, "code": render(function), "bug_line": None, "mutation": None})
            rows.append({**base, "code": render(planted), "bug_line": function[i][0], "mutation": name, "original": function[i][1].strip(), "changed": planted[i][1].strip()})
            kinds[name] += 1
            taken += 1
    print(split, "pairs", len(rows) // 2, "repos", len({r["repo"] for r in rows}), dict(kinds.most_common()))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cap", type=int, default=150)
    parser.add_argument("--val-cap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    parse = parsers()
    licences = {}
    for name in ("train.jsonl", "val.jsonl"):
        for line in (PUBLIC / "dataset" / name).read_text().splitlines():
            row = json.loads(line)
            licences[row["repo"]] = row["licence"]
    train_repos = sorted(r for r in licences if r not in VAL_REPOS and (CLONES / r.replace("/", "__")).exists())
    rows = build("train", train_repos, licences, args.train_cap, rng, parse) + build("val", VAL_REPOS, licences, args.val_cap, rng, parse)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(args.out, args.out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
