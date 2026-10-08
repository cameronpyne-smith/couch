import math
from dataclasses import dataclass

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.weight.T + self.bias


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        )


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.project_back(self.non_linearity(self.expand(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        C = config.n_embedding_dimension
        assert C % config.n_head == 0, "C must divide evenly by n_head"
        self.qkv_projection = Linear(C, 3 * C)
        self.output_projection = Linear(C, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        head_dim = C // self.config.n_head
        qkv = self.qkv_projection(x)
        query, key, value = qkv.split(C, dim=-1)
        query = query.view(B, T, self.config.n_head, head_dim).transpose(1, 2)
        key = key.view(B, T, self.config.n_head, head_dim).transpose(1, 2)
        value = value.view(B, T, self.config.n_head, head_dim).transpose(1, 2)
        out = causal_attention(query, key, value)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.output_projection(out)


class LayerNorm(nn.Module):
    def __init__(self, n_features, eps=1e-5):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(n_features))
        self.bias = nn.Parameter(torch.zeros(n_features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalised = (x - mean) / torch.sqrt(variance + self.eps)
        return (normalised * self.gain) + self.bias

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = CausalSelfAttention(config) 
        self.mlp = MLP(config)
        self.layer_norm_1 = LayerNorm(config.n_embedding_dimension)
        self.layer_norm_2 = LayerNorm(config.n_embedding_dimension)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))
        return x


def causal_attention(query, key, value):
    head_dim = query.shape[-1]
    scores = (
        query @ key.transpose(-2, -1) / math.sqrt(head_dim)
    )  # transpose turns (T, d) into (d, T). divide by sqrt(d) to 'normalise' the dot product so weighting isn't so skewed
    T = query.shape[-2]
    mask = torch.tril(
        torch.ones(T, T, device=query.device)
    )  # tril = triangle lower, zeros above diagonal
    scores = scores.masked_fill(mask == 0, float("-inf"))  # -inf so exp = 0 weight
    scores = scores - scores.amax(
        dim=-1, keepdim=True
    )  # subtract largest score from the row, "per row" is dim=-1
    weights = torch.exp(scores)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights @ value


if __name__ == "__main__":
    print(ModelConfig())
    print(ModelConfig(n_layer=2))
    x = torch.randn(2, 8, 384)
    mlp = MLP(ModelConfig())
    print(mlp(x).shape)
    print(sum(p.numel() for p in mlp.parameters()))
    print(torch.allclose(GELU()(x), F.gelu(x, approximate="tanh")))

    q, k, v = torch.randn(3, 2, 6, 8, 64)
    print(
        torch.allclose(
            causal_attention(q, k, v),
            F.scaled_dot_product_attention(q, k, v, is_causal=True),
            atol=1e-6,
        )
    )

    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[..., 4:, :] += 1
    k2[..., 4:, :] += 1
    v2[..., 4:, :] += 1

    print(
        torch.allclose(
            causal_attention(q, k, v)[..., :4, :],
            causal_attention(q2, k2, v2)[..., :4, :],
            atol=1e-6,
        )
    )

    csa = CausalSelfAttention(ModelConfig())
    print(sum(p.numel() for p in csa.parameters()))
    print(csa(x).shape)
    x2 = x.clone()
    x2[:, 4:, :] += 1
    print(torch.allclose(csa(x)[..., :4, :], csa(x2)[..., :4, :], atol=1e-6))

    print(torch.allclose(LayerNorm(384)(x), F.layer_norm(x, (384,)), atol=1e-5))

    block = Block(ModelConfig())
    print(block(x).shape)
    print(sum(p.numel() for p in block.parameters()))
    print(((block(x) - x).abs().mean()) / x.abs().mean() * 100)

