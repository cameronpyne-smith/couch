import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune.first_token import auc
from optimiser import AdamW


class Probe(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def load(path):
    data = torch.load(path)
    states = data["states"]
    examples = data["examples"]
    correct = [float(example["kind"] == "real") for example in examples]
    correct = torch.tensor(correct, dtype=torch.float32)
    repos = [example["repo"] for example in examples] # Specific for this example, could be any suspect group
    return states, correct, repos


def prompt_lengths(path):
    with open(path) as file:
        return [len(json.loads(line)["prompt"]) for line in file]


def group_pairs(y, groups):
    return [
        (real, clean)
        for real in range(len(groups))
        for clean in range(len(groups))
        if y[real] == 1 and y[clean] != 1 and groups[real] == groups[clean]
    ]


def pair_auc(scores, pairs):
    return sum(scores[real] > scores[clean] for real, clean in pairs) / len(pairs)


def within_group_auc(scores, y, groups):
    wins = 0.0
    pairs = 0
    for group in set(groups):
        mask = torch.tensor([g == group for g in groups])
        real = scores[mask & (y == 1)].tolist()
        clean = scores[mask & (y != 1)].tolist()
        count = len(real) * len(clean)
        if count == 0:
            continue
        wins += auc(real, clean) * count
        pairs += count
    return wins / pairs


def train_probe(probe, x, y, steps, learning_rate, log=False):
    optimiser = AdamW(probe.parameters(), learning_rate, weight_decay=0.0)

    for step in range(steps):
        loss = F.binary_cross_entropy_with_logits(probe(x), y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        if log and step % 20 == 0:
            print(f"step: {step} - loss: {loss.item()}")


def evaluate(probe, x, y):
    with torch.no_grad():
        scores = probe(x)
        real = scores[y == 1]
        clean = scores[y != 1]
        return auc(real.tolist(), clean.tolist())


def cross_validate(x, y, groups, folds, steps, learning_rate, values=None):
    names = sorted(set(groups))
    fold_of = {name: index % folds for index, name in enumerate(names)}
    row_fold = torch.tensor([fold_of[group] for group in groups])
    scores = torch.zeros(len(groups))

    for fold in range(folds):
        held_out = row_fold == fold
        x_train = x[~held_out]
        x_held_out = x[held_out]
        if values is not None:
            groups_train = [group for group, held in zip(groups, held_out) if not held]
            x_train = centre_within(x_train, groups_train)
            values_train = centre_within(values[~held_out], groups_train)
            slopes = trend_slopes(x_train, values_train)
            x_train = subtract_trend(x_train, values_train, slopes)
            x_held_out = subtract_trend(x_held_out, values[held_out], slopes)

        probe = Probe(x.shape[1])
        train_probe(probe, x_train, y[~held_out], steps, learning_rate)
        with torch.no_grad():
            scores[held_out] = probe(x_held_out)

    return scores


def sweep(folder):
    states, y, repos = load(folder / "train.pt")
    states_val, y_val, repos_val = load(folder / "val.pt")
    states_val_full, y_val_full, repos_val_full = load(folder / "val-full.pt")

    for layer in range(states.shape[1]):
        probe = Probe(states.shape[2])
        train_probe(probe, states[:, layer], y, 200, 0.001)
        with torch.no_grad():
            within = within_group_auc(probe(states_val[:, layer]), y_val, repos_val)
            within_full = within_group_auc(probe(states_val_full[:, layer]), y_val_full, repos_val_full)
        print(f"layer: {layer} - held-out within-repository auc: {within:.3f} - val-full within-repository auc: {within_full:.3f}")


def length_control(folder, prompts, layers):
    states, y, _ = load(folder / "train.pt")
    states_val_full, y_val_full, repos = load(folder / "val-full.pt")
    lengths = prompt_lengths(prompts)
    assert len(lengths) == len(repos)

    pairs = group_pairs(y_val_full, repos)
    longer = [(real, clean) for real, clean in pairs if lengths[real] > lengths[clean]]
    shorter = [(real, clean) for real, clean in pairs if lengths[real] <= lengths[clean]]
    print(f"{folder.name} - pairs - all: {len(pairs)} - buggy longer: {len(longer)} - buggy shorter: {len(shorter)}")
    print(f"{folder.name} - length auc - all: {pair_auc(lengths, pairs):.3f}")

    for layer in layers:
        probe = Probe(states.shape[2])
        train_probe(probe, states[:, layer], y, 200, 0.001)
        with torch.no_grad():
            scores = probe(states_val_full[:, layer]).tolist()
        print(f"{folder.name} - layer: {layer} - probe auc - all: {pair_auc(scores, pairs):.3f} - buggy longer: {pair_auc(scores, longer):.3f} - buggy shorter: {pair_auc(scores, shorter):.3f}")


def centre_within(x, groups):
    copy = x.clone()
    for group in set(groups):
        mask = torch.tensor([g == group for g in groups])
        average = (mask.float() @ copy) / mask.sum()
        copy[mask] -= average
    return copy

def trend_slopes(x, values):
    v = values - values.mean()
    top = v @ x
    bottom = v @ v
    slopes = top / bottom
    return slopes

def column_stats(x):
    mean = torch.mean(x, dim=0)
    std = torch.std(x, dim=0).clamp(min=1e-6)
    return mean, std

def standardise(x, mean, std):
    return (x - mean) / std

def subtract_trend(x, values, slopes):
    v = values - values.mean()
    trend = torch.outer(v, slopes)
    return x - trend


def remove_trend(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    slopes = trend_slopes(x, values)
    return subtract_trend(x, values, slopes)



def erasure_control(folder, prompts, layers):
    states, y, repos = load(folder / "train.pt")
    values = torch.tensor(prompt_lengths(prompts), dtype=torch.float32).log()
    assert len(values) == len(repos)
    pairs = group_pairs(y, repos)

    length_only = values[:, None]
    plain = cross_validate(length_only, y, repos, 5, 2000, 0.05)
    erased = cross_validate(length_only, y, repos, 5, 2000, 0.05, values)
    print(f"{folder.name} - length column - plain: {pair_auc(plain.tolist(), pairs):.3f} - erased: {pair_auc(erased.tolist(), pairs):.3f}")

    for layer in layers:
        plain = cross_validate(states[:, layer], y, repos, 5, 200, 0.001)
        erased = cross_validate(states[:, layer], y, repos, 5, 200, 0.001, values)
        print(f"{folder.name} - layer: {layer} - held-out-repository auc - plain: {pair_auc(plain.tolist(), pairs):.3f} - repository and length erased: {pair_auc(erased.tolist(), pairs):.3f}")


if __name__ == "__main__":
    activations = Path("~/training-runs/public-bugs/activations").expanduser()
    prompts = Path("~/training-data/code/public-bugs/dataset/sft/train.jsonl").expanduser()
    erasure_control(activations / "base", prompts, [1, 24, 48, 64])
