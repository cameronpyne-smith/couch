import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from finetune.dpo import dpo_loss, pair_log_probs, pairs_from_samples, reference_log_probs, sequence_log_probs
from finetune.sft import IGNORE, collate, compute_loss


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6, 4)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class StandIn(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Body()
        self.lm_head = nn.Linear(4, 6)


def test_stand_in_shapes():
    torch.manual_seed(0)
    ids = torch.tensor([[1, 2, 3]])
    stand_in = StandIn()
    output = stand_in.model(input_ids=ids, attention_mask=None)
    hidden = output.last_hidden_state
    scores = stand_in.lm_head(hidden)
    assert hidden.shape == (1, 3, 4)
    assert scores.shape == (1, 3, 6)


def test_rows_add_up_to_batch():
    torch.manual_seed(0)
    long_ids = [1, 2, 3, 4, 5]
    long_labels = [IGNORE, IGNORE, 3, 4, 5]
    short_ids = [2, 1, 4]
    short_labels = [IGNORE, 1, 4]
    batch = [(long_ids, long_labels), (short_ids, short_labels)]
    input_ids, attention_mask, labels = collate(batch, 0)
    stand_in = StandIn()
    output = sequence_log_probs(stand_in, input_ids, attention_mask, labels)
    output_2 = compute_loss(stand_in, input_ids, attention_mask, labels)
    assert output.shape == (2,)
    assert torch.allclose(output.sum(), -output_2, atol=1e-5)


def test_row_is_unchanged_by_padding():
    torch.manual_seed(0)
    long_ids = [1, 2, 3, 4, 5]
    long_labels = [IGNORE, IGNORE, 3, 4, 5]
    short_ids = [2, 1, 4]
    short_labels = [IGNORE, 1, 4]
    batch = [(long_ids, long_labels), (short_ids, short_labels)]
    ids_1, attention_mask_1, labels_1 = collate(
        [(long_ids, long_labels), (short_ids, short_labels)], 0
    )
    ids_2, attention_mask_2, labels_2 = collate([(short_ids, short_labels)], 0)
    stand_in = StandIn()
    output_1 = sequence_log_probs(stand_in, ids_1, attention_mask_1, labels_1)
    output_2 = sequence_log_probs(stand_in, ids_2, attention_mask_2, labels_2)
    assert torch.allclose(output_1[1], output_2[0], atol=1e-5)


@pytest.fixture
def base():
    return torch.tensor([-5.0, -8.0])

def says(verdict, number):
    return {"verdict": verdict, "text": f"says {verdict} {number}"}


def test_loss_is_log_2_when_policy_equals_reference(base):
    assert dpo_loss(base, base, base, base, 1).item() == pytest.approx(math.log(2))


def test_loss_falls_when_chosen_gains_and_rises_when_rejected_gains(base):
    start = dpo_loss(base, base, base, base, 1)
    assert dpo_loss(base + 1, base, base, base, 1) < start
    assert dpo_loss(base, base + 1, base, base, 1) > start


def test_loss_depends_only_on_change_from_reference(base):
    before = dpo_loss(base + 1, base, base, base, 1)
    after = dpo_loss(base + 1 + 3, base, base + 3, base, 1)
    assert before.item() != pytest.approx(math.log(2))
    assert after.item() == pytest.approx(before.item())


def test_two_right_two_wrong_gives_two_pairs():
    samples = [says("break", 1), says("no break", 1), says("no break", 2), says("break", 2)]
    assert pairs_from_samples(samples, "break") == [("says break 1", "says no break 1"), ("says break 2", "says no break 2")]
    assert pairs_from_samples(samples, "no break") == [("says no break 1", "says break 1"), ("says no break 2", "says break 2")]


def test_unanimous_samples_give_no_pairs():
    samples = [says("break", number) for number in range(1, 5)]
    assert pairs_from_samples(samples, "break") == []
    assert pairs_from_samples(samples, "no break") == []


def test_three_right_one_wrong_gives_one_pair():
    samples = [says("break", 1), says("break", 2), says("no break", 1), says("break", 3)]
    assert pairs_from_samples(samples, "break") == [("says break 1", "says no break 1")]


def test_sample_without_a_verdict_is_in_no_pair():
    samples = [says("none", 1), says("break", 1), says("no break", 1)]
    assert pairs_from_samples(samples, "break") == [("says break 1", "says no break 1")]


def test_reference_log_probs_scores_each_pair_without_gradients():
    torch.manual_seed(0)
    stand_in = StandIn()
    pairs = [
        (([1, 2, 3, 4], [IGNORE, IGNORE, 3, 4]), ([1, 2, 5], [IGNORE, IGNORE, 5])),
        (([3, 1], [IGNORE, 1]), ([3, 4, 2, 5], [IGNORE, 4, 2, 5])),
    ]
    result = reference_log_probs(stand_in, pairs, 0, "cpu")
    assert result.shape == (2, 2)
    for index, pair in enumerate(pairs):
        assert torch.allclose(result[index], pair_log_probs(stand_in, pair, 0, "cpu"))
    assert not result.requires_grad
