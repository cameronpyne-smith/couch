import argparse
from pathlib import Path

import torch

from generate import generate, load_model
from model import cross_entropy
from optimiser import AdamW, clip_gradients
from train import estimate_loss, get_batch, get_learning_rate, make_directory, log_losses
from finetune.lora import apply_lora


def get_training_and_validation_data(tokeniser, new_training_data_path):
    file_content = new_training_data_path.read_text()
    encoded_data = torch.tensor(tokeniser.encode(file_content), dtype=torch.long)
    training_data_length = int(0.9 * len(encoded_data))
    training_data = encoded_data[:training_data_length]
    validation_data = encoded_data[training_data_length:]
    return training_data, validation_data

def save_lora_adapter(model, path):
    file_name = "adapter.pt"
    file_path = Path(path) / file_name
    adapter = {name: tensor for name, tensor in model.state_dict().items() if name.endswith((".A", ".B"))}

    torch.save(adapter, file_path)




if __name__ == "__main__":


    max_steps = 5000
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", default=max_steps, type=int)
    args = parser.parse_args()
    max_steps = args.max_steps



    eval_interval = 100
    batch_size = 64
    device = "cuda"
    directory = make_directory("runs")
    max_norm = 1.0
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
        model, new_validation, model.config, batch_size, device 
    )
    old_training_validation_loss = estimate_loss(
        model, old_validation, model.config, batch_size, device
    )
    print(f"Old loss: {old_training_validation_loss}")
    print(f"New loss: {new_training_validation_loss}")
    lora_params = apply_lora(model, 2, 2, ["qkv_projection", "output_projection", "expand", "project_back"])
    model.to(device)
    print(f"Trainable parameters: {sum([p.numel() for p in lora_params])}")
    print(f"Total: {sum([p.numel() for p in model.parameters()])}")
    optimiser = AdamW(lora_params)

    stats = []
    lowest_validation_average = float("inf")
    for step in range(max_steps):
        lr = get_learning_rate(step, max_steps=max_steps)
        if step % eval_interval == 0:
            validation_average = estimate_loss(
                model, new_validation, model.config, batch_size, device
            )
            training_average = estimate_loss(
                model, new_training, model.config, batch_size, device
            )
            original_average = estimate_loss(
                model, old_validation, model.config, batch_size, device
            )
            if validation_average < lowest_validation_average:
                lowest_validation_average = validation_average
                save_lora_adapter(model, directory)
            stats.append(
                {
                    "step": step,
                    "train": training_average,
                    "val": validation_average,
                    "old_val": original_average,
                    "lr": lr,
                }
            )
            print(
                f"Step {step}, training: {training_average:.3f} validation: {validation_average:.3f} original validation: {original_average:.3f}"
            )
            print(f"\n--- step {step} sample ---")
            print(generate(model, tokeniser, "\n", 200))
            log_losses(stats, directory)
        x, y = get_batch(new_training, model.config.block_size, batch_size, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        clip_gradients(lora_params, max_norm)
        optimiser.learning_rate = lr
        optimiser.step()
        optimiser.zero_grad()
