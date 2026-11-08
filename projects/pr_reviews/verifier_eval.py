import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from finetune.first_token import auc
from finetune.lora import load_lora_adapter
from finetune.sft import IGNORE, MODEL, encode, load_examples, load_model
from projects.pr_reviews.dataset import OUTPUT, VERDICTS


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


def report(name, predictions, golds, losses, scores):
    pairs = list(zip(predictions, golds))
    tp = sum(p == "invalid" and g == "invalid" for p, g in pairs)
    fp = sum(p == "invalid" and g == "valid" for p, g in pairs)
    fn = sum(p == "valid" and g == "invalid" for p, g in pairs)
    tn = sum(p == "valid" and g == "valid" for p, g in pairs)
    recall_invalid = tp / (tp + fn) if tp + fn else float("nan")
    recall_valid = tn / (tn + fp) if tn + fp else float("nan")
    precision_invalid = tp / (tp + fp) if tp + fp else float("nan")
    ranking = auc([s for s, g in zip(scores, golds) if g == "invalid"], [s for s, g in zip(scores, golds) if g == "valid"])
    print(
        f"{name}: n {len(pairs)} accuracy {(tp + tn) / len(pairs):.3f}"
        f" recall invalid {recall_invalid:.3f} ({tp}/{tp + fn}) recall valid {recall_valid:.3f} ({tn}/{tn + fp})"
        f" balanced {(recall_invalid + recall_valid) / 2:.3f} precision invalid {precision_invalid:.3f}"
        f" word loss {sum(losses) / len(losses):.3f} auc {ranking:.3f}"
    )


def evaluate(name, model, splits, tokeniser, device):
    for split, examples in splits.items():
        predictions, losses, scores = predict(model, examples, tokeniser, device)
        report(f"{name} {split}", predictions, [example["answer"] for example in examples], losses, scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=OUTPUT)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--rank", default=16, type=int)
    parser.add_argument("--alpha", default=32, type=int)
    args = parser.parse_args()

    device = "cuda"
    tokeniser = AutoTokenizer.from_pretrained(MODEL)
    model, _ = load_model(args.rank, args.alpha, device)
    splits = {name: load_examples(args.data / f"{name}.jsonl") for name in ("val", "smoke")}

    evaluate("base", model, splits, tokeniser, device)
    load_lora_adapter(model, args.adapter)
    evaluate("adapter", model, splits, tokeniser, device)
    valid_fraction = sum(example["answer"] == "valid" for example in splits["val"]) / len(splits["val"])
    print(f"always valid val: accuracy {valid_fraction:.3f} balanced 0.500")
