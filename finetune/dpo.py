import torch
import torch.nn.functional as F

from finetune.sft import IGNORE, encode, to_device


def sequence_log_probs(model, input_ids, attention_mask, labels):
    hidden = model.model(
        input_ids=input_ids, attention_mask=attention_mask
    ).last_hidden_state
    targets = labels[:, 1:]
    keep = targets != IGNORE
    logits = model.lm_head(hidden[:, :-1][keep]).float()
    losses = F.cross_entropy(logits, targets[keep], reduction="none")
    grid = torch.zeros_like(targets, dtype=losses.dtype)
    grid[keep] = losses
    summed_loss = grid.sum(dim=1)
    return -summed_loss


# policy = model being trained, reference = frozen starting model
def dpo_loss(
    policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta
):
    chosen_gain = policy_chosen - reference_chosen
    rejected_gain = policy_rejected - reference_rejected
    margin = chosen_gain - rejected_gain
    x = beta * margin
    loss = -F.logsigmoid(x)
    return loss.mean()


def pairs_from_samples(samples, correct):
    right = []
    wrong = []
    for sample in samples:
        if sample["verdict"] == "none":
            continue

        if sample["verdict"] == correct:
            right.append(sample["text"])
        else:
            wrong.append(sample["text"])

    return list(zip(right, wrong))


def encode_pair(row, tokeniser):
    chosen = encode(row | {"answer": row["chosen"]}, tokeniser)
    rejected = encode(row | {"answer": row["rejected"]}, tokeniser)
    return chosen, rejected


def pair_log_probs(model, pair, pad_id, device):
    input_ids, attention_mask, labels = to_device(list(pair), pad_id, device)
    return sequence_log_probs(model, input_ids, attention_mask, labels)

def reference_log_probs(model, pairs, pad_id, device):
    results = []
    with torch.no_grad():
        for pair in pairs:
            result = pair_log_probs(model, pair, pad_id, device)
            results.append(result.cpu())
    return torch.stack(results)
