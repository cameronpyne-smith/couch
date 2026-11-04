import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune.lora import apply_lora

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA = Path.home() / "training-data" / "code" / "pr-reviews-verifier"
IGNORE = -100
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_examples(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def encode(example, tokeniser):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["prompt"]},
    ]
    prompt_ids = tokeniser.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)
    answer_ids = tokeniser(example["answer"], add_special_tokens=False).input_ids + [tokeniser.eos_token_id]
    return prompt_ids + answer_ids, [IGNORE] * len(prompt_ids) + answer_ids


def make_batches(encoded, batch_size, seed):
    ordered = sorted(encoded, key=lambda pair: len(pair[0]))
    batches = [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]
    random.Random(seed).shuffle(batches)
    return batches


def collate(batch, pad_id):
    width = max(len(ids) for ids, _ in batch)
    input_ids = [ids + [pad_id] * (width - len(ids)) for ids, _ in batch]
    attention_mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch]
    labels = [row + [IGNORE] * (width - len(row)) for _, row in batch]
    return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(labels)


def load_model(rank, alpha, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    trainable = apply_lora(model, rank, alpha, PROJECTIONS)
    return model, trainable


if __name__ == "__main__":
    tokeniser = AutoTokenizer.from_pretrained(MODEL)
    train = load_examples(DATA / "train.jsonl")

    ids, labels = encode(train[0], tokeniser)
    masked = sum(label == IGNORE for label in labels)
    print("answer:", repr(tokeniser.decode([label for label in labels if label != IGNORE])))
    print("prompt tail:", repr(tokeniser.decode(ids[:masked])[-40:]))

    encoded = [encode(example, tokeniser) for example in train]
    lengths = sorted(len(ids) for ids, _ in encoded)
    print("tokens: median", lengths[len(lengths) // 2], "max", lengths[-1])

    batches = make_batches(encoded, 4, 0)
    cells = real = 0
    for batch in batches:
        input_ids, attention_mask, labels = collate(batch, tokeniser.pad_token_id)
        cells += input_ids.numel()
        real += int(attention_mask.sum())
    print("batches", len(batches), "pad fraction", round(1 - real / cells, 3))
    for batch in (batches[0], batches[-1]):
        input_ids, attention_mask, labels = collate(batch, tokeniser.pad_token_id)
        print(tuple(input_ids.shape), "real", int(attention_mask.sum()), "answer tokens", int((labels != IGNORE).sum()))

    model, trainable = load_model(rank=16, alpha=32)
    total = sum(p.numel() for p in model.parameters())
    print("params", total / 1e6, "M, trainable", sum(p.numel() for p in trainable) / 1e6, "M")
    print("weights", torch.cuda.memory_allocated() / 2**30, "GB")

    input_ids, attention_mask, labels = (t.to("cuda") for t in collate(batches[0], tokeniser.pad_token_id))
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()
    first = model.model.layers[0].self_attn.q_proj
    print("loss", loss.item(), "A grad", first.A.grad is not None, "base grad", first.base.weight.grad)
    print("peak", torch.cuda.max_memory_allocated() / 2**30, "GB")
