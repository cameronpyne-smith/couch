from functools import partial

import pytest
import torch

from generate import load_model
from model import GPT, ModelConfig, cross_entropy
from tokeniser import Tokeniser
from train import get_batch, get_learning_rate, load_data, save_checkpoint, train

WARM_UP_STEPS = 10
MAX_STEPS = 110
MAX_LEARNING_RATE = 1.0
MIN_LEARNING_RATE = 0.1


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
    training, _, tokeniser = load_data()
    config = ModelConfig(
        vocab_size=tokeniser.vocab_size,
        n_layer=2,
        n_head=2,
        n_embedding_dimension=64,
        block_size=64,
    )
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


learning_rate = partial(
    get_learning_rate,
    warm_up_steps=WARM_UP_STEPS,
    max_steps=MAX_STEPS,
    max_learning_rate=MAX_LEARNING_RATE,
    min_learning_rate=MIN_LEARNING_RATE,
)


def test_lr_starts_linear():
    assert learning_rate(0) == MAX_LEARNING_RATE / WARM_UP_STEPS


def test_lr_peaks_at_warm_up():
    assert learning_rate(WARM_UP_STEPS) == pytest.approx(MAX_LEARNING_RATE)


def test_lr_ends_at_min_lr():
    assert learning_rate(MAX_STEPS) == MIN_LEARNING_RATE
    assert learning_rate(MAX_STEPS + 10) == MIN_LEARNING_RATE


def test_decay_midpoint_is_average_of_max_and_min():
    midpoint = (WARM_UP_STEPS + MAX_STEPS) // 2
    assert learning_rate(midpoint) == pytest.approx(
        (MAX_LEARNING_RATE + MIN_LEARNING_RATE) / 2
    )


def test_lr_never_increases_after_warm_up():
    rates = [learning_rate(step) for step in range(WARM_UP_STEPS, MAX_STEPS + 10)]
    assert all(later <= earlier for earlier, later in zip(rates, rates[1:]))


def test_train_saves(tmp_path):
    torch.manual_seed(0)
    max_steps = 3
    config = ModelConfig(
        vocab_size=65,
        n_layer=2,
        n_head=2,
        n_embedding_dimension=64,
        block_size=64,
    )
    assert not any(tmp_path.iterdir())
    train(
        config=config,
        max_steps=max_steps,
        eval_interval=1,
        device="cpu",
        output_dir=tmp_path,
    )
    assert len(list(tmp_path.glob("*/losses.json"))) == 1
    assert len(list(tmp_path.glob("*/checkpoint.pt"))) == 1
    checkpoint = torch.load(list(tmp_path.glob("*/checkpoint.pt"))[0])
    assert checkpoint["step"] < max_steps


def test_save_checkpoint_then_load_model_returns_input(tmp_path):
    config = ModelConfig(
        vocab_size=4, block_size=8, n_layer=1, n_head=1, n_embedding_dimension=8
    )
    model = GPT(config)
    tokeniser = Tokeniser("abc\n")
    save_checkpoint(model, config, tokeniser, 7, tmp_path)
    loaded_model, loaded_tokeniser, step = load_model(
        tmp_path / "checkpoint.pt", device="cpu"
    )
    assert step == 7
    assert loaded_tokeniser.chars == tokeniser.chars
    assert loaded_model.config == model.config
    loaded_dict = loaded_model.state_dict()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded_dict[name])
