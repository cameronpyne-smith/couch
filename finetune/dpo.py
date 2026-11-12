import torch
import torch.nn.functional as F


IGNORE = -100


def sequence_log_probs(model, input_ids, attention_mask, labels):
    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    targets = labels[:, 1:]
    keep = targets != IGNORE
    logits = model.lm_head(hidden[:, :-1][keep]).float()
    losses = F.cross_entropy(logits, targets[keep], reduction="none")
    grid = torch.zeros_like(targets, dtype=losses.dtype)
    grid[keep] = losses
    summed_loss = grid.sum(dim=1)
    return -summed_loss
