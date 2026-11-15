import json

import pytest
import torch
from safetensors.torch import load_file

from finetune.peft_export import export, peft_config, peft_state


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


def small_adapter():
    return {
        "model.layers.0.q_proj.A": torch.randn(4, 3),
        "model.layers.0.q_proj.B": torch.randn(5, 4),
        "model.layers.1.out_proj.A": torch.randn(4, 6),
        "model.layers.1.out_proj.B": torch.randn(2, 4),
    }


def test_config_reads_rank_and_targets_from_the_tensors():
    config = peft_config(small_adapter(), "org/base", 8)
    assert config["r"] == 4
    assert config["lora_alpha"] == 8
    assert config["target_modules"] == ["out_proj", "q_proj"]
    assert config["base_model_name_or_path"] == "org/base"


def test_config_refuses_mixed_ranks():
    adapter = small_adapter() | {"model.layers.2.q_proj.A": torch.randn(7, 3)}
    with pytest.raises(ValueError, match="one rank"):
        peft_config(adapter, "org/base", 8)


def test_export_writes_tensors_and_config_that_read_back(tmp_path):
    adapter = small_adapter()
    torch.save(adapter, tmp_path / "adapter.pt")
    export(tmp_path, tmp_path / "out", "org/base", 8)
    written = load_file(tmp_path / "out" / "adapter_model.safetensors")
    expected = peft_state(adapter)
    assert written.keys() == expected.keys()
    assert all(torch.equal(written[name], expected[name]) for name in expected)
    assert json.loads((tmp_path / "out" / "adapter_config.json").read_text()) == peft_config(adapter, "org/base", 8)
