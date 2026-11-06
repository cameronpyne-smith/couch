import argparse

import torch
from transformers import AutoTokenizer

from finetune.sft import DATA, IGNORE, MODEL, RUNS, encode, load_examples, load_model

VERDICTS = ("valid", "invalid")


def load_adapter(model, path):
    adapter = torch.load(path)
    report = model.load_state_dict(adapter, strict=False)
    assert not report.unexpected_keys, report.unexpected_keys


def predict(model, examples, tokeniser, device):
    ids_of = {word: tokeniser(word, add_special_tokens=False).input_ids[0] for word in VERDICTS}
    predictions = []
    losses = []
    scores = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            ids, labels = encode(example, tokeniser)
            prompt = ids[: labels.count(IGNORE)]
            logits = model(input_ids=torch.tensor([prompt], device=device)).logits[0, -1]
            predictions.append(max(VERDICTS, key=lambda word: logits[ids_of[word]].item()))
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            losses.append(-log_probs[ids_of[example["answer"]]].item())
            scores.append((log_probs[ids_of["invalid"]] - log_probs[ids_of["valid"]]).item())
    return predictions, losses, scores


def ranking(scores, golds):
    positives = [s for s, g in zip(scores, golds) if g == "invalid"]
    negatives = [s for s, g in zip(scores, golds) if g == "valid"]
    if not positives or not negatives:
        return float("nan")
    ahead = sum(p > n for p in positives for n in negatives)
    return ahead / (len(positives) * len(negatives))


def report(name, predictions, golds, losses, scores):
    pairs = list(zip(predictions, golds))
    tp = sum(p == "invalid" and g == "invalid" for p, g in pairs)
    fp = sum(p == "invalid" and g == "valid" for p, g in pairs)
    fn = sum(p == "valid" and g == "invalid" for p, g in pairs)
    tn = sum(p == "valid" and g == "valid" for p, g in pairs)
    recall_invalid = tp / (tp + fn) if tp + fn else float("nan")
    recall_valid = tn / (tn + fp) if tn + fp else float("nan")
    precision_invalid = tp / (tp + fp) if tp + fp else float("nan")
    print(
        f"{name}: n {len(pairs)} accuracy {(tp + tn) / len(pairs):.3f}"
        f" recall invalid {recall_invalid:.3f} ({tp}/{tp + fn}) recall valid {recall_valid:.3f} ({tn}/{tn + fp})"
        f" balanced {(recall_invalid + recall_valid) / 2:.3f} precision invalid {precision_invalid:.3f}"
        f" word loss {sum(losses) / len(losses):.3f} auc {ranking(scores, golds):.3f}"
    )


def evaluate(name, model, splits, tokeniser, device):
    for split, examples in splits.items():
        predictions, losses, scores = predict(model, examples, tokeniser, device)
        report(f"{name} {split}", predictions, [example["answer"] for example in examples], losses, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=RUNS / "20261104-013859" / "adapter.pt")
    parser.add_argument("--rank", default=16, type=int)
    parser.add_argument("--alpha", default=32, type=int)
    args = parser.parse_args()

    device = "cuda"
    tokeniser = AutoTokenizer.from_pretrained(MODEL)
    model, _ = load_model(args.rank, args.alpha, device)
    splits = {name: load_examples(DATA / f"{name}.jsonl") for name in ("val", "smoke")}

    evaluate("base", model, splits, tokeniser, device)
    load_adapter(model, args.adapter)
    evaluate("adapter", model, splits, tokeniser, device)
    valid_fraction = sum(example["answer"] == "valid" for example in splits["val"]) / len(splits["val"])
    print(f"always valid val: accuracy {valid_fraction:.3f} balanced 0.500")
