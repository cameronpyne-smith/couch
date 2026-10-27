import argparse

import torch

from model import GPT, ModelConfig
from tokeniser import Tokeniser


def load_model(file_path: str, device="cuda"):
    d = torch.load(file_path, weights_only=False)
    step = d["step"]
    config = ModelConfig(**d["config"])
    model = GPT(config)
    model.load_state_dict(d["state"])
    model = model.to(device).eval()
    tokeniser = Tokeniser("".join(d["tokeniser"]))
    return (model, tokeniser, step)


def generate(model, tokeniser, prompt, max_new_tokens, temperature=0.8, top_k=40):
    encoded_prompt = tokeniser.encode(prompt)
    device = next(model.parameters()).device
    ids = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[:, -model.config.block_size :]
            logits = model(context)  # (1, T, 65)
            scores = logits[:, -1, :]  # (1, 65)
            scores = scores / temperature
            options, _ = torch.topk(scores, top_k)
            threshold = options[:, -1:]
            mask = scores < threshold
            scores[mask] = float("-inf")
            probs = torch.softmax(scores, dim=-1)
            winner = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, winner], dim=1)
    return tokeniser.decode(ids[0].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="To be or not", type=str)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-k", default=40, type=int)
    parser.add_argument("--max-new-tokens", default=500, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model, tokeniser, _ = load_model(args.checkpoint)
    print(
        generate(
            model,
            tokeniser,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
        )
    )
