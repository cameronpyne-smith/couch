import math
from pathlib import Path

import torch

from tokeniser import Tokeniser

DEFAULT_DATA_PATH = Path.home() / "training-data" / "text" / "shakespeare.txt"


def load_data(file_path=DEFAULT_DATA_PATH):
    file_content = file_path.read_text()
    tokeniser = Tokeniser(file_content)
    encoded_data = torch.tensor(tokeniser.encode(file_content), dtype=torch.long)
    training_data_length = int(0.9 * len(encoded_data))
    training_data = encoded_data[:training_data_length]
    validation_data = encoded_data[training_data_length:]
    return training_data, validation_data, tokeniser


def get_batch(data, block_size, batch_size, device):
    offsets = torch.randint(len(data) - block_size - 1, (batch_size,))
    inputs = []
    targets = []
    for offset in offsets:
        inputs.append(data[offset : offset + block_size])
        targets.append(data[offset + 1 : offset + 1 + block_size])
    x = torch.stack(inputs).to(device)
    y = torch.stack(targets).to(device)
    return x, y


def get_learning_rate(
    step, warm_up_steps=100, max_steps=5000, max_learning_rate=1e-3, min_learning_rate=1e-4
) -> float:
    if step < warm_up_steps: # warmup
        return max_learning_rate * (step + 1) / warm_up_steps
    if step >= max_steps: # finished
        return min_learning_rate
    else: # cosine decay
        progress = (step - warm_up_steps) / (max_steps - warm_up_steps)
        return min_learning_rate + (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress)) / 2
