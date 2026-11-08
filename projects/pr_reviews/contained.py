import json
import random

from finetune.ollama import ask, boolean_schema
from projects.pr_reviews.generator import OUTPUT

CONTAINED = OUTPUT.with_name(OUTPUT.name + "-contained")
RULE = (
    "You are shown one function and a code review finding about it. Decide whether a reviewer who sees only this function text, "
    "and nothing else from the repository, could verify the finding. It is contained when the problem it describes is visible in "
    "the code shown. It is not contained when checking it needs other files, the pull request diff, callers, tests, or "
    "declarations defined elsewhere. "
    'Reply with JSON only: {"contained": true} or {"contained": false}.'
)
SCHEMA = boolean_schema("contained")


def contained(prompt, finding, rule=RULE, prompt_chars=6000, num_ctx=None):
    return ask(rule, f"Function:\n{prompt[:prompt_chars]}\n\nFinding:\n{finding[:1500]}", SCHEMA, num_ctx=num_ctx)["contained"]


def key(example):
    return f"{example['repo']}#{example['number']}:{example['answer']}"


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def label(splits, path, asker=contained):
    verdicts = {row["key"]: row["contained"] for row in load(path)} if path.exists() else {}
    with path.open("a") as handle:
        for name, examples in splits.items():
            reals = [e for e in examples if e["kind"] == "real"]
            for index, example in enumerate(reals):
                if key(example) in verdicts:
                    continue
                verdicts[key(example)] = verdict = asker(example["prompt"], example["answer"])
                handle.write(json.dumps({"key": key(example), "contained": verdict}) + "\n")
                handle.flush()
                if index % 50 == 0:
                    print(f"  {name} {index}/{len(reals)} contained so far {sum(verdicts.values())}/{len(verdicts)}", flush=True)
    return verdicts


def build(splits, verdicts):
    for name, examples in splits.items():
        real = [e for e in examples if e["kind"] == "real" and verdicts[key(e)]]
        clean = random.Random(0).sample([e for e in examples if e["kind"] == "clean"], len(real))
        kept = [e for e in examples if e in real or e in clean]
        with (CONTAINED / f"{name}.jsonl").open("w") as handle:
            for example in kept:
                handle.write(json.dumps(example) + "\n")
        print(f"{name}: real {len(real)}/{sum(e['kind'] == 'real' for e in examples)} clean {len(clean)} kept {len(kept)}")
        if name == "val":
            answers = load(OUTPUT / "base" / "Qwen3.8-27B" / "answers-base.jsonl")
            assert all(a["gold"] == e["answer"] and a["number"] == e["number"] for a, e in zip(answers, examples))
            target = CONTAINED / "base" / "Qwen3.8-27B" / "answers-base.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w") as handle:
                for answer, example in zip(answers, examples):
                    if example in real or example in clean:
                        handle.write(json.dumps(answer) + "\n")


if __name__ == "__main__":
    CONTAINED.mkdir(exist_ok=True)
    splits = {name: load(OUTPUT / f"{name}.jsonl") for name in ("train", "val")}
    verdicts = label(splits, CONTAINED / "verdicts.jsonl")
    examples = [e for s in splits.values() for e in s if e["kind"] == "real"]
    for wanted in (True, False):
        print("=" * 80, "contained" if wanted else "not contained")
        for e in random.Random(0).sample([e for e in examples if verdicts[key(e)] == wanted], 3):
            print("-> " + e["answer"][:300])
    build(splits, verdicts)
