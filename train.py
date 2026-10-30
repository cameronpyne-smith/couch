import argparse
import dataclasses
import json
import math
from datetime import datetime
from pathlib import Path

import torch

from generate import generate
from model import GPT, ModelConfig, cross_entropy
from optimiser import AdamW, clip_gradients
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
    step,
    warm_up_steps=100,
    max_steps=5000,
    max_learning_rate=1e-3,
    min_learning_rate=1e-4,
) -> float:
    if step < warm_up_steps:  # warmup
        return max_learning_rate * (step + 1) / warm_up_steps
    if step >= max_steps:  # finished
        return min_learning_rate
    else:  # cosine decay
        progress = (step - warm_up_steps) / (max_steps - warm_up_steps)
        return (
            min_learning_rate
            + (max_learning_rate - min_learning_rate)
            * (1 + math.cos(math.pi * progress))
            / 2
        )


def estimate_loss(model, data, config, batch_size, device, n_batches=20):
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        for _ in range(n_batches):
            x, y = get_batch(data, config.block_size, batch_size, device)
            logits = model(x)
            loss = cross_entropy(logits, y)
            total_loss += loss.item()
        model.train()
        return total_loss / n_batches


def make_directory(output_dir):
    directory = Path(output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def log_losses(stats, path):
    file_name = "losses.json"
    file_path = Path(path) / file_name
    file_path.write_text(json.dumps(stats, indent=2))


def save_checkpoint(model, config, tokeniser, step, path):
    file_name = "checkpoint.pt"
    file_path = Path(path) / file_name
    obj = {
        "state": model.state_dict(),
        "config": dataclasses.asdict(config),
        "tokeniser": tokeniser.chars,
        "step": step,
    }
    torch.save(obj, file_path)


def train(
    max_steps=5000,
    batch_size=64,
    device="cuda",
    eval_interval=100,
    output_dir="./runs",
    config=None,
    input_data_path=DEFAULT_DATA_PATH,
):
    directory = make_directory(output_dir)
    data_path = Path(input_data_path)
    training, validation, tokeniser = load_data(data_path)
    if config is None:
        config = ModelConfig(vocab_size=tokeniser.vocab_size)
    max_norm = 1.0
    model = GPT(config).to(device)
    optimiser = AdamW(model.parameters())
    stats = []
    lowest_validation_average = float("inf")
    for step in range(max_steps):
        lr = get_learning_rate(step, max_steps=max_steps)
        if step % eval_interval == 0:
            validation_average = estimate_loss(
                model, validation, config, batch_size, device
            )
            training_average = estimate_loss(
                model, training, config, batch_size, device
            )
            if validation_average < lowest_validation_average:
                lowest_validation_average = validation_average
                save_checkpoint(model, config, tokeniser, step, directory)
            stats.append(
                {
                    "step": step,
                    "train": training_average,
                    "val": validation_average,
                    "lr": lr,
                }
            )
            print(
                f"Step {step}, training: {training_average:.3f} validation: {validation_average:.3f}"
            )
            print(f"\n--- step {step} sample ---")
            print(generate(model, tokeniser, "\n", 200))
        x, y = get_batch(training, config.block_size, batch_size, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        clip_gradients(model.parameters(), max_norm)
        optimiser.learning_rate = lr
        optimiser.step()
        optimiser.zero_grad()
    log_losses(stats, directory)
    return model, tokeniser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-path", default=DEFAULT_DATA_PATH, type=str)
    parser.add_argument("--output-dir", default="./runs", type=str)
    parser.add_argument("--max-steps", default=5000, type=int)
    args = parser.parse_args()
    train(input_data_path=args.input_data_path, output_dir=args.output_dir, max_steps=args.max_steps)
