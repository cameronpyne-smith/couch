import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

PREFIX = "base_model.model."
ENDINGS = {".A": ".lora_A.weight", ".B": ".lora_B.weight"}


def peft_name(name):
    for ending, replacement in ENDINGS.items():
        if name.endswith(ending):
            return PREFIX + name.removesuffix(ending) + replacement
    raise ValueError(f"not a LoRA tensor name: {name}")


def peft_state(adapter):
    return {peft_name(name): tensor for name, tensor in adapter.items()}


def peft_config(adapter, model, alpha):
    ranks = {tensor.shape[0] for name, tensor in adapter.items() if name.endswith(".A")}
    if len(ranks) != 1:
        raise ValueError(f"expected one rank across the adapter, found {sorted(ranks)}")
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model,
        "r": ranks.pop(),
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "target_modules": sorted({name.rsplit(".", 2)[-2] for name in adapter}),
    }


def export(run_dir, output_dir, model, alpha):
    adapter = torch.load(run_dir / "adapter.pt", map_location="cpu", weights_only=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(peft_state(adapter), output_dir / "adapter_model.safetensors", metadata={"format": "pt"})
    (output_dir / "adapter_config.json").write_text(json.dumps(peft_config(adapter, model, alpha), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--alpha", type=int, default=32)
    args = parser.parse_args()
    export(args.run_dir, args.output_dir, args.model, args.alpha)
