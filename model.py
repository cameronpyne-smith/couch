from dataclasses import dataclass

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embedding_dimension: int = 384

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias 

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.expand = Linear(
            config.n_embedding_dimension, config.n_embedding_dimension * 4
        )
        self.non_linearity = GELU()
        self.project_back = Linear(
            4 * config.n_embedding_dimension, config.n_embedding_dimension
        )

    def forward(self, x):
        return self.project_back(self.non_linearity(self.expand(x)))


if __name__ == "__main__":
    print(ModelConfig())
    print(ModelConfig(n_layer=2))
    x = torch.randn(2, 8, 384)
    mlp = MLP(ModelConfig())
    print(mlp(x).shape)
    print(sum(p.numel() for p in mlp.parameters()))
    print(torch.allclose(GELU()(x), F.gelu(x, approximate="tanh")))
