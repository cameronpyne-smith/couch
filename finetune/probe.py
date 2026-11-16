from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    return states[:, layer], correct

def train_probe(probe, x, y, steps, learning_rate):
    optimiser = AdamW(probe.parameters(), learning_rate, weight_decay=0.0)

    for step in range(steps):
        loss = F.binary_cross_entropy_with_logits(probe(x), y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        if step % 20 == 0:
            print(f"step: {step} - loss: {loss.item()}")
    


if __name__ == "__main__":
    x, y = load_layer(
        Path("~/training-runs/public-bugs/activations/base/train.pt").expanduser(), 32
    )
    probe = Probe(x.shape[1])
    train_probe(probe, x, y, 200, 0.001)

