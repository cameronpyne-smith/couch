import torch

from model import GPT, ModelConfig
from tokeniser import Tokeniser

def load_model(file_path: str, device = "cuda"):
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
            context = ids[:, -model.config.block_size:]
            logits = model(context) # (1, T, 65)
            scores = logits[:, -1, :] # (1, 65)
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
    torch.manual_seed(0)
    model, tokeniser, step = load_model("runs/20261021-171312/checkpoint.pt")
    print(generate(model, tokeniser, "To be or not", 200, 0.8, 40))
    assert step == 900

