import torch

from types import SimpleNamespace
from torch import nn
from finetune.sft import IGNORE, collate, compute_loss
from finetune.dpo import sequence_log_probs 


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
    ids_1, attention_mask_1, labels_1 = collate([(long_ids, long_labels), (short_ids, short_labels)], 0)
    ids_2, attention_mask_2, labels_2 = collate([(short_ids, short_labels)], 0)
    stand_in = StandIn()
    output_1 = sequence_log_probs(stand_in, ids_1, attention_mask_1, labels_1)
    output_2 = sequence_log_probs(stand_in, ids_2, attention_mask_2, labels_2)
    assert torch.allclose(output_1[1], output_2[0], atol=1e-5)


