from pathlib import Path

import torch
import torch.nn as nn

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


if __name__ == "__main__":
    x, y = load_layer(
        Path("~/training-runs/public-bugs/activations/base/train.pt").expanduser(), 32
    )
    print(f"x.shape: {x.shape}, y.shape: {y.shape}, y.sum: {y.sum()}")
    probe = Probe(x.shape[1])
    print(f"{probe(x).shape} | {probe(x)[:5]}")
