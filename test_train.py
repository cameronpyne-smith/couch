from model import ModelConfig
from train import get_batch

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
