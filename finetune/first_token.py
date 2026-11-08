import torch


def probability(logits, token_id):
    return torch.softmax(logits.float(), dim=-1)[token_id].item()


def auc(higher, lower):
    if not higher or not lower:
        return float("nan")
    return sum(h > l for h in higher for l in lower) / (len(higher) * len(lower))


def matched_threshold(scores, answered):
    ordered = sorted(scores)
    if answered <= 0:
        return 0.0
    if answered >= len(ordered):
        return 1.0
    return (ordered[answered - 1] + ordered[answered]) / 2
