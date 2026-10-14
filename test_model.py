import pytest
import torch
import torch.nn.functional as F

from model import (
    GELU,
    GPT,
    MLP,
    Block,
    CausalSelfAttention,
    Embedding,
    LayerNorm,
    ModelConfig,
    causal_attention,
)


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig()


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(2, 8, 384)


@pytest.fixture
def qkv():
    torch.manual_seed(0)
    return torch.randn(3, 2, 6, 8, 64)


@pytest.fixture
def idx():
    torch.manual_seed(0)
    return torch.randint(0, 65, (2, 8))


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


def test_mlp_output_shape(config, x):
    assert MLP(config)(x).shape == (2, 8, 384)


def test_mlp_parameter_count(config):
    assert parameter_count(MLP(config)) == 1181568


def test_gelu_matches_torch_tanh_approximation(x):
    assert torch.allclose(GELU()(x), F.gelu(x, approximate="tanh"), atol=1e-6)


def test_causal_attention_matches_torch(qkv):
    q, k, v = qkv
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(causal_attention(q, k, v), expected, atol=1e-6)


def test_causal_attention_ignores_future_positions(qkv):
    q, k, v = qkv
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[..., 4:, :] += 1
    k2[..., 4:, :] += 1
    v2[..., 4:, :] += 1
    before = causal_attention(q, k, v)[..., :4, :]
    after = causal_attention(q2, k2, v2)[..., :4, :]
    assert torch.allclose(before, after, atol=1e-6)


def test_attention_module_parameter_count(config):
    assert parameter_count(CausalSelfAttention(config)) == 591360


def test_attention_module_output_shape(config, x):
    assert CausalSelfAttention(config)(x).shape == (2, 8, 384)


def test_attention_module_ignores_future_positions(config, x):
    attention = CausalSelfAttention(config)
    x2 = x.clone()
    x2[:, 4:, :] += 1
    assert torch.allclose(attention(x)[:, :4, :], attention(x2)[:, :4, :], atol=1e-6)


def test_layer_norm_matches_torch(x):
    assert torch.allclose(LayerNorm(384)(x), F.layer_norm(x, (384,)), atol=1e-5)


def test_block_output_shape(config, x):
    assert Block(config)(x).shape == (2, 8, 384)


def test_block_parameter_count(config):
    assert parameter_count(Block(config)) == 1774464


def test_block_is_near_identity_at_init(config, x):
    relative_change = (Block(config)(x) - x).abs().mean() / x.abs().mean()
    assert relative_change < 0.5


def test_embeddings_parameter_count(config: ModelConfig):
    assert (
        parameter_count(Embedding(config.vocab_size, config.n_embedding_dimension))
        == 24960
    )


def test_embeddings_output_shape(config, idx):
    assert Embedding(config.vocab_size, config.n_embedding_dimension)(idx).shape == (
        2,
        8,
        384,
    )


def test_embeddings_output_matches_torch(config, idx):
    embedding = Embedding(config.vocab_size, config.n_embedding_dimension)
    assert torch.allclose(embedding(idx), F.embedding(idx, embedding.weight), atol=1e-6)


def test_gpt_parameter_count(config):
    assert parameter_count(GPT(config)) == 10770816

def test_gpt_logits_shape(config, idx):
    assert GPT(config)(idx).shape == (2, 8, 65)

def test_gpt_logits_tied_weights(config):
    gpt = GPT(config)
    assert gpt.token_embedding.weight is gpt.output_head.weight

def test_gpt_logits_ignore_future_positions(config, idx):
    gpt = GPT(config)
    idx2 = idx.clone()
    idx2[:, 4:] = (idx[:, 4:] + 1) % 65
    logits1 = gpt(idx)
    logits2 = gpt(idx2)
    assert torch.allclose(logits1[:, :4, :], logits2[:, :4, :], atol=1e-6)


