from pathlib import Path

file_path = Path.home() / "training-data" / "text" / "shakespeare.txt"
file_content = file_path.read_text()

chars = sorted(set(file_content))
char_to_id_map = {c: i for i, c in enumerate(chars)}
vocab_size = len(chars)


def encode(text: str) -> list[int]:
    return [char_to_id_map[c] for c in text]


def decode(numbers: list[int]) -> str:
    return "".join(chars[i] for i in numbers)


if __name__ == "__main__":
    assert decode(encode(file_content)) == file_content
