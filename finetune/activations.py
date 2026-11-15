import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from finetune.lora import load_lora_adapter
from finetune.sft import IGNORE, MODEL, encode, load_examples, load_model

KEYS = ("kind", "repo", "number", "path", "line")


def prompt_ids(example, tokeniser):
    ids, labels = encode(example, tokeniser)
    return ids[: labels.count(IGNORE)]


def last_token_states(model, ids, device):
    output = model.model(input_ids=torch.tensor([ids], device=device), output_hidden_states=True)
    return torch.stack([layer[0, -1] for layer in output.hidden_states]).float().cpu()


def extract(name, model, tokeniser, examples, device, path):
    model.eval()
    states = []
    started = time.time()
    with torch.no_grad():
        for index, example in enumerate(examples):
            states.append(last_token_states(model, prompt_ids(example, tokeniser), device))
            if index % 50 == 0:
                print(f"  {name} {index}/{len(examples)} {time.time() - started:.0f} s", flush=True)
    states = torch.stack(states)
    if not torch.isfinite(states).all():
        raise ValueError(f"{name}: non-finite activations")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"states": states, "examples": [{key: example[key] for key in KEYS} for example in examples]}, path)
    print(f"{name}: saved {tuple(states.shape)} (examples, layers + 1, hidden) to {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--adapter", type=Path, help="adapter run directory; a second pass with it loaded is saved under its name")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--bits", type=int, default=16, choices=(16, 4))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    device = "cuda"
    tokeniser = AutoTokenizer.from_pretrained(args.model)
    model, _ = load_model(args.rank, args.alpha, device, args.model, args.bits)
    splits = {split: load_examples(args.data / f"{split}.jsonl")[: args.limit] for split in args.splits}
    for split, examples in splits.items():
        extract(f"base {split}", model, tokeniser, examples, device, args.output_dir / "base" / f"{split}.pt")
    if args.adapter:
        load_lora_adapter(model, args.adapter / "adapter.pt")
        for split, examples in splits.items():
            extract(f"adapter {split}", model, tokeniser, examples, device, args.output_dir / f"adapter-{args.adapter.name}" / f"{split}.pt")
    print(f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
