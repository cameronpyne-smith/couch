import copy

import pytest
import torch

from model import GPT, ModelConfig, cross_entropy
from optimiser import AdamW, clip_gradients
from train import get_batch, load_data


@pytest.fixture
def two_comparison_models():
    torch.manual_seed(0)
    training, _, tokeniser = load_data()
    config = ModelConfig(
        vocab_size=tokeniser.vocab_size,
        n_layer=2,
        n_head=2,
        n_embedding_dimension=64,
        block_size=64,
        dropout=0.0
    )
    model_1 = GPT(config)
    model_2 = copy.deepcopy(model_1)
    return training, config, model_1, model_2


def test_adamw_matches_torch_adamw(two_comparison_models):
    training, config, model_1, model_2 = two_comparison_models
    optimiser_1 = AdamW(
        model_1.parameters(),
        learning_rate=0.001,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.01,
    )
    optimiser_2 = torch.optim.AdamW(
        model_2.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )

    for _ in range(20):
        x, y = get_batch(training, config.block_size, 32, "cpu")

        for model, optimiser in ((model_1, optimiser_1), (model_2, optimiser_2)):
            loss = cross_entropy(model(x), y)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

    for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-5)


def test_clip_gradient():
    a = torch.nn.Parameter(torch.zeros(2))
    b = torch.nn.Parameter(torch.zeros(2))
    a.grad = torch.tensor([3.0, 0.0])
    b.grad = torch.tensor([0.0, 4.0])
    max_norm = 1.0
    norm = clip_gradients([a, b], max_norm)
    assert norm == 5.0  # sqrt(9 + 16) = 5, scale = 1/5 = 0.2
    assert torch.allclose(a.grad, torch.tensor([0.6, 0.0]))
    assert torch.allclose(b.grad, torch.tensor([0.0, 0.8]))


def test_clip_gradients_matches_torch(two_comparison_models):
    max_norm = 1.0
    training, config, model_1, model_2 = two_comparison_models
    x, y = get_batch(training, config.block_size, 32, "cpu")
    for model in (model_1, model_2):
        cross_entropy(model(x), y).backward()
    norm_model = clip_gradients(model_1.parameters(), max_norm)
    norm_reference = torch.nn.utils.clip_grad_norm_(model_2.parameters(), max_norm)
    assert norm_model == pytest.approx(norm_reference.item())
    for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
        assert p1.grad is not None and p2.grad is not None
        assert torch.allclose(p1.grad, p2.grad)
