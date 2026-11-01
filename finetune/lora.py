import torch
import torch.nn as nn


# Linear computes x @ W.T + b, full fine-tuning would update every number, LorRA freezes W and learns a correction B @ A beside it.
class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        base.requires_grad_(False)
        self.base = base
        out_features = self.base.weight.shape[0]
        in_features = self.base.weight.shape[1]
        self.A = nn.Parameter(torch.randn(rank, in_features) / in_features**0.5)
        self.B = nn.Parameter(torch.zeros((out_features, rank)))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
