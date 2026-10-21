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
     
if __name__ == "__main__":
    model, tokeniser, step = load_model("runs/20261021-171312/checkpoint.pt")
    print(step)
    assert step == 900

