import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATA = Path.home() / "training-data" / "code" / "pr-reviews-verifier"
IGNORE = -100


def load_examples(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def encode(example, tokeniser):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["prompt"]},
    ]
    prompt_ids = tokeniser.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)
    answer_ids = tokeniser(example["answer"], add_special_tokens=False).input_ids + [tokeniser.eos_token_id]
    return prompt_ids + answer_ids, [IGNORE] * len(prompt_ids) + answer_ids


if __name__ == "__main__":
    tokeniser = AutoTokenizer.from_pretrained(MODEL)
    train = load_examples(DATA / "train.jsonl")

    ids, labels = encode(train[0], tokeniser)
    masked = sum(label == IGNORE for label in labels)
    print("answer:", repr(tokeniser.decode([label for label in labels if label != IGNORE])))
    print("prompt tail:", repr(tokeniser.decode(ids[:masked])[-40:]))

    lengths = sorted(len(encode(example, tokeniser)[0]) for example in train)
    print("tokens: median", lengths[len(lengths) // 2], "max", lengths[-1])
