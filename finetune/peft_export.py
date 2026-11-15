PREFIX = "base_model.model."
ENDINGS = {".A": ".lora_A.weight", ".B": ".lora_B.weight"}


def peft_name(name):
    for ending, replacement in ENDINGS.items():
        if name.endswith(ending):
            return PREFIX + name.removesuffix(ending) + replacement
    raise ValueError(f"not a LoRA tensor name: {name}")


def peft_state(adapter):
    return {peft_name(name): tensor for name, tensor in adapter.items()}
