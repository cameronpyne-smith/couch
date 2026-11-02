import argparse
from pathlib import Path
from collections import Counter

import torch

from tokeniser import Tokeniser


SHAKESPEARE = Path.home() / "training-data" / "text" / "shakespeare.txt"
PRIDE_AND_PREJUDICE = Path.home() / "training-data" / "text" / "pride-and-prejudice-raw.txt"

def find_missing_chars(old_training_data_path, new_training_data_path):
    old_training_data = old_training_data_path.read_text()
    new_training_data = new_training_data_path.read_text()

    counts = Counter(new_training_data)
    vocab = set(Tokeniser(old_training_data).chars)
    missing = {char: n for char, n in counts.items() if char not in vocab}

    for char, n in sorted(missing.items(), key=lambda item: item[1], reverse=True):
        print(repr(char), n)

    return missing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-data-path", default=SHAKESPEARE, type=str)
    parser.add_argument("--new-data-path", default=PRIDE_AND_PREJUDICE, type=str)
    args = parser.parse_args()

    find_missing_chars(Path(args.old_data_path), Path(args.new_data_path))
