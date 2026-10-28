import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

def plot():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    directory = Path(args.directory)
    file = directory / "losses.json"
    losses = json.loads(file.read_text())
    steps = [row["step"] for row in losses]
    train = [row["train"] for row in losses]
    val = [row["val"] for row in losses]

    plt.plot(steps, train, label="train")
    plt.plot(steps, val, label="val")
    min_loss_index = min(range(len(val)), key=val.__getitem__)
    plt.axvline(steps[min_loss_index], linestyle="--")

    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(directory / "loss.png")

if __name__ == "__main__":
    plot()

