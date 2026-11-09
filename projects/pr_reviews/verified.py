from collections import Counter

from projects.pr_reviews.contained import load
from projects.pr_reviews.context import SUBSET
from projects.pr_reviews.dataset import write_jsonl
from projects.pr_reviews.verdicts import load_verdicts

VERIFIED = SUBSET.with_name(SUBSET.name.replace("-contained", "-verified"))
KEEP_REAL = ("contained", "needs_context")


def latest_labels(rows, set_name):
    labels = {}
    for row in rows:
        if row["set"] == set_name:
            split = row["split"].split("-n")[0]
            labels[(split, int(row["index"]), row["subject"])] = row
    return labels


def label_for(labels, split, index, kind):
    label = labels.get((split, index, "label"))
    if kind == "clean":
        for (s, i, subject), row in labels.items():
            if (s, i) == (split, index) and subject != "label" and row["verdict"] == "real" and (label is None or label["verdict"] == "clean"):
                label = {**row, "verdict": "has_bug", "note": f"model finding confirmed by {row['reader']}: {row['note']}"}
    return label


def build(split, examples, labels):
    kept, found, sidecar = [], [], []
    for index, example in enumerate(examples):
        label = label_for(labels, split, index, example["kind"])
        if label is None:
            continue
        verdict = label["verdict"]
        sidecar.append({"index": index, "kind": example["kind"], "verdict": verdict, "severity": label.get("severity"), "note": label["note"], "evidence": label.get("evidence", ""), "reader": label["reader"]})
        if example["kind"] == "real" and verdict in KEEP_REAL:
            kept.append({**example, "contained": verdict == "contained"})
        elif example["kind"] == "clean" and verdict == "clean":
            kept.append({**example, "contained": True})
        elif verdict == "has_bug":
            found.append({**example, "defect": label["note"], "severity": label.get("severity"), "evidence": label.get("evidence", "")})
    return kept, found, sidecar


if __name__ == "__main__":
    labels = latest_labels(load_verdicts(), SUBSET.name)
    for split in ("train", "val"):
        examples = load(SUBSET / f"{split}.jsonl")
        kept, found, sidecar = build(split, examples, labels)
        write_jsonl(VERIFIED / f"{split}.jsonl", kept)
        write_jsonl(VERIFIED / f"{split}-labels.jsonl", sidecar)
        write_jsonl(VERIFIED / f"{split}-defects-on-clean.jsonl", found)
        before = Counter(e["kind"] for e in examples)
        after = Counter(e["kind"] for e in kept)
        verdicts = Counter((row["kind"], row["verdict"]) for row in sidecar)
        print(f"{split}: labelled {len(sidecar)}/{len(examples)}; real {before['real']} -> {after['real']}, clean {before['clean']} -> {after['clean']}")
        print("   ", dict(sorted(verdicts.items())))
        print(f"    contained among kept real: {sum(e['contained'] for e in kept if e['kind'] == 'real')}/{after['real']}")
