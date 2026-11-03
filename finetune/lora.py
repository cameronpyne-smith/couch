import torch
import torch.nn as nn

from model import Linear


# Linear computes x @ W.T + b, full fine-tuning would update every number, LorRA freezes W and learns a correction B @ A beside it.
class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        base.requires_grad_(False)
        self.base = base
        out_features = self.base.weight.shape[0]
        in_features = self.base.weight.shape[1]
        self.A = nn.Parameter(torch.randn(rank, in_features, device=base.weight.device) / in_features**0.5)
        self.B = nn.Parameter(torch.zeros((out_features, rank), device=base.weight.device))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
    
    # x @ A.T @ B.T = x @ (B @ A).T
    def merge(self):
        with torch.no_grad():
            self.base.weight += (self.B @ self.A) * self.scale
        return self.base


def apply_lora(model, rank, alpha, match):
    model.requires_grad_(False) # Freeze model

    def is_match(name, module):
        return isinstance(module, Linear) and name.rsplit(".", 1)[-1] in match

    pairs = [(name, module) for name, module in model.named_modules() if is_match(name, module)]
    for name, module in pairs:
        parent_path, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, attribute, LoRALinear(module, rank, alpha))
    return [p for p in model.parameters() if p.requires_grad]

