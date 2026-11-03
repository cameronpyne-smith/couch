from pathlib import Path

import torch

from finetune.lora import LoRALinear, apply_lora, merge_lora
from generate import generate, load_model
from train import save_checkpoint


def check_model(checkpoint_path, adapter_path):
    model, tokeniser, steps = load_model(checkpoint_path)
    apply_lora(model, 2, 2, ["qkv_projection", "output_projection", "expand", "project_back"])
    adapter = torch.load(adapter_path)
    report = model.load_state_dict(adapter, strict=False)
    assert not report.unexpected_keys, report.unexpected_keys
    return model, tokeniser, steps

def merged_model_and_wrapped_model_generate_same_output(checkpoint_path, adapter_path):
    model, tokeniser, step = check_model(checkpoint_path, adapter_path)
    torch.manual_seed(0)
    before = generate(model, tokeniser, "\n", 200)
    merge_lora(model)
    torch.manual_seed(0)
    after = generate(model, tokeniser, "\n", 200)
    assert before == after
    assert sum(isinstance(m, LoRALinear) for m in model.modules()) == 0
    save_checkpoint(model, model.config, tokeniser, step, Path(adapter_path).parent)


if __name__ == "__main__":
    CHECKPOINT_PATH = "runs/20261028-004026/checkpoint.pt"
    ADAPTER_PATH = "runs/20261103-005354/adapter.pt"
    merged_model_and_wrapped_model_generate_same_output(CHECKPOINT_PATH, ADAPTER_PATH)
