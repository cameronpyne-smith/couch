import torch
import pytest

from model import Linear
from finetune.lora import LoRALinear

@pytest.fixture
def lora():
    return LoRALinear(Linear(4, 3), rank=2, alpha=2)

@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(1, 4)


def test_A_and_B_have_rank_shapes(lora):
    assert lora.A.shape == (2, 4)
    assert lora.B.shape == (3, 2)

def test_only_A_and_B_are_trainable(lora):
    assert not lora.base.weight.requires_grad
    assert sum(p.requires_grad for p in lora.parameters()) == 2

def test_output_equals_base_at_init_and_moves_with_B(lora, x):
    assert torch.equal(lora.base(x), lora(x))
    with torch.no_grad():
        lora.B.fill_(1.0)
    assert not torch.equal(lora.base(x), lora(x))

def test_training_only_moves_A_and_B(lora, x):
    lora(x).sum().backward()
    assert lora.base.weight.grad is None
    assert lora.B.grad is not None
