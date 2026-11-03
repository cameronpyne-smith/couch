import json
import subprocess
from collections import Counter
from pathlib import Path


GOLDEN = Path.home() / "code" / "reviewdo" / "bench" / "golden"
EXPORT = Path.home() / "training-data" / "code" / "pr-reviews"
CLONES = Path.home() / "code" / "remundo"
VERDICTS = ("valid", "invalid")


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
        check=True,
    )
    return result.stdout


if __name__ == "__main__":
    records = load_records()
    print("by verdict", Counter(r["finding"]["verdict"] for r in records))
    print("by label source", Counter(r["finding"]["labelled"].split(":")[0] for r in records))
    print("without diff", sum(r["diff"] is None for r in records))

    example = next(r for r in records if r["repo"].endswith("Pricing.Employee") and r["number"] == 13)
    lines = read_file_at_commit(example).splitlines()
    print(example["finding"]["line"], lines[example["finding"]["line"] - 1])
