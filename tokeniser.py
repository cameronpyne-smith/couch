class Tokeniser:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_id_map = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id_map[c] for c in text]

    def decode(self, numbers: list[int]) -> str:
        return "".join(self.chars[i] for i in numbers)
