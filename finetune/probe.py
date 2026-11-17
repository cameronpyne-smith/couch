from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune.first_token import auc
from optimiser import AdamW


class Probe(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def load_layer(path, layer):
    data = torch.load(path)
    states = data["states"]
    examples = data["examples"]
    correct = [float(example["kind"] == "real") for example in examples]
    correct = torch.tensor(correct, dtype=torch.float32)
    repos = [example["repo"] for example in examples] # Specific for this example, could be any suspect group
    return states[:, layer], correct, repos


def within_group_auc(scores, y, groups):
    wins = 0.0
    pairs = 0
    for group in set(groups):
        mask = torch.tensor([g == group for g in groups])
        real = scores[mask & (y == 1)].tolist()
        clean = scores[mask & (y != 1)].tolist()
        count = len(real) * len(clean)
        if count == 0:
            continue
        wins += auc(real, clean) * count
        pairs += count
    return wins / pairs


def train_probe(probe, x, y, steps, learning_rate):
    optimiser = AdamW(probe.parameters(), learning_rate, weight_decay=0.0)

    for step in range(steps):
        loss = F.binary_cross_entropy_with_logits(probe(x), y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        if step % 20 == 0:
            print(f"step: {step} - loss: {loss.item()}")


def evaluate(probe, x, y):
    with torch.no_grad():
        scores = probe(x)
        real = scores[y == 1]
        clean = scores[y != 1]
        return auc(real.tolist(), clean.tolist())


if __name__ == "__main__":
    x, y, repos = load_layer(
        Path("~/training-runs/public-bugs/activations/base/train.pt").expanduser(), 32
    )
    probe = Probe(x.shape[1])
    train_probe(probe, x, y, 200, 0.001)

    x_val, y_val, repos_val = load_layer(
        Path("~/training-runs/public-bugs/activations/base/val.pt").expanduser(), 32
    )
    with torch.no_grad():
        within = within_group_auc(probe(x_val), y_val, repos_val)
    print(f"train auc: {evaluate(probe, x, y):.3f} held-out auc: {evaluate(probe, x_val, y_val):.3f} held-out within-repository auc: {within:.3f}")
