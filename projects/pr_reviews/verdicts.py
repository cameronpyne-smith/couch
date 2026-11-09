import json
from collections import Counter, defaultdict

from projects.pr_reviews.packets import REVIEW


def load_verdicts():
    rows = []
    for path in sorted(REVIEW.glob("verdicts-*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["reader"] = path.stem.removeprefix("verdicts-")
                rows.append(row)
    return rows


def manifest(row):
    path = REVIEW / row["set"] / row["split"] / "manifest.jsonl"
    return {m["index"]: m for m in (json.loads(l) for l in path.read_text().splitlines())}


def summarise(rows):
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["set"], row["split"], int(row["index"]))][row["subject"]] = row
    print("latest line per function and subject wins:", sum(len(v) for v in by_key.values()), "verdicts")
    kinds = {}
    for key in by_key:
        kinds.setdefault(key[:2], manifest({"set": key[0], "split": key[1]}))
    for (set_name, split), group in sorted({k[:2]: None for k in by_key}.items()):
        items = {k: v for k, v in by_key.items() if k[:2] == (set_name, split)}
        print(f"== {set_name} / {split}: {len(items)} functions judged")
        for kind in ("clean", "real"):
            subset = {k: v for k, v in items.items() if kinds[(set_name, split)][k[2]]["kind"] == kind}
            if not subset:
                continue
            print(f"-- {kind} ({len(subset)})")
            labels = Counter(v["label"]["verdict"] for v in subset.values() if "label" in v)
            print("   label:", dict(labels))
            for subject in sorted({s for v in subset.values() for s in v if s != "label"}):
                verdicts = Counter(v[subject]["verdict"] for v in subset.values() if subject in v)
                severities = Counter(v[subject].get("severity") for v in subset.values() if subject in v and v[subject]["verdict"] == "real")
                print(f"   {subject}: {dict(verdicts)} severities of real {dict(severities)}")
            if kind == "clean":
                print("   defects found on clean functions:")
                for key, v in sorted(subset.items()):
                    hits = [s for s in v if v[s]["verdict"] in ("real", "has_bug")]
                    if hits:
                        who = v[hits[0]]
                        m = kinds[(set_name, split)][key[2]]
                        print(f"     {key[2]:04d} {m['repo'].split('/')[-1]}#{m['number']} {m.get('path', '')} [{', '.join(hits)}] {who.get('severity')}: {who['note'][:160]}")
            else:
                both = [v for v in subset.values() if "base" in v and "adapter" in v]
                agree = Counter((v["base"]["verdict"] == "real", v["adapter"]["verdict"] == "real") for v in both)
                print("   (base real, adapter real):", dict(agree))


if __name__ == "__main__":
    rows = load_verdicts()
    print(len(rows), "verdict lines from", Counter(r["reader"] for r in rows))
    summarise(rows)
