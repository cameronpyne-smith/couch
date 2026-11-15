import pytest
import torch

from finetune.peft_export import peft_state


def test_names_follow_peft_layout():
    adapter = {"model.layers.0.out_proj.A": torch.zeros(2, 3), "model.layers.0.out_proj.B": torch.zeros(3, 2)}
    assert list(peft_state(adapter)) == [
        "base_model.model.model.layers.0.out_proj.lora_A.weight",
        "base_model.model.model.layers.0.out_proj.lora_B.weight",
    ]


def test_tensors_are_passed_through_uncopied():
    adapter = {"model.layers.0.out_proj.A": torch.zeros(2, 3), "model.layers.0.out_proj.B": torch.zeros(3, 2)}
    assert all(new is old for new, old in zip(peft_state(adapter).values(), adapter.values()))


def test_unknown_ending_is_refused():
    with pytest.raises(ValueError, match="out_proj.bias"):
        peft_state({"model.layers.0.out_proj.bias": torch.zeros(2)})
