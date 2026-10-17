from model import ModelConfig, GPT, cross_entropy
from train import get_batch, load_data

import pytest
import torch

@pytest.fixture
def batch():
    training_data = torch.arange(1000)
    x, y = get_batch(training_data, ModelConfig.block_size, 64, "cpu")
    return x, y

@pytest.fixture
def x(batch):
    x, _ = batch
    return x

@pytest.fixture
def y(batch):
    _, y = batch
    return y

def test_get_batch_shape(x, y):
    assert x.shape == y.shape == (64, 256)

def test_get_batch_targets_are_next_token(x, y):
    assert torch.equal(y, x + 1)

def test_loss_decreases_with_gradient_descent():
    torch.manual_seed(0)
    training, _, tokenizer = load_data()
    config = ModelConfig(vocab_size=tokenizer.vocab_size, n_layer=2, n_head=2, n_embedding_dimension=64, block_size=64)
    gpt = GPT(config)
    losses = []
    for _ in range(20):
        x, y = get_batch(training, config.block_size, 32, "cpu")
        logits = gpt(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        with torch.no_grad():
            for p in gpt.parameters():
                assert p.grad is not None
                p -= 0.1 * p.grad
                p.grad = None
        losses.append(loss.item())

    assert losses[-1] < losses[0]
    assert losses[-1] < 3.6
        


