from pathlib import Path

import torch

from generate import load_model
from model import ModelConfig
from train import estimate_loss


def get_training_and_validation_data(tokeniser, new_training_data_path):
    file_content = new_training_data_path.read_text()
    encoded_data = torch.tensor(tokeniser.encode(file_content), dtype=torch.long)
    training_data_length = int(0.9 * len(encoded_data))
    training_data = encoded_data[:training_data_length]
    validation_data = encoded_data[training_data_length:]
    return training_data, validation_data


if __name__ == "__main__":
    model, tokeniser, step = load_model("runs/20261028-004026/checkpoint.pt")
    pride_and_prejudice = (
        Path.home() / "training-data" / "text" / "pride-and-prejudice.txt"
    )
    shakespeare = Path.home() / "training-data" / "text" / "shakespeare.txt"
    new_training, new_validation = get_training_and_validation_data(
        tokeniser, pride_and_prejudice
    )
    old_training, old_validation = get_training_and_validation_data(
        tokeniser, shakespeare
    )
    new_training_validation_loss = estimate_loss(
        model, new_validation, model.config, 64, "cuda"
    )
    old_training_validation_loss = estimate_loss(
        model, old_validation, model.config, 64, "cuda"
    )
    print(f"Old loss: {old_training_validation_loss}")
    print(f"New loss: {new_training_validation_loss}")
