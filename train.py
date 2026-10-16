from tokenizer import encode, decode, file_content
from model import ModelConfig

import torch

def load_data():
    encoded_data = torch.tensor(encode(file_content), dtype=torch.long)
    training_data_length = int(0.9 * len(encoded_data))
    training_data = encoded_data[:training_data_length]
    validation_data = encoded_data[training_data_length:]
    return training_data, validation_data

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

if __name__ == "__main__":
    training, validation = load_data()
    x, y = get_batch(training, ModelConfig.block_size, 64, "cuda")
    print(x.shape == y.shape == (64, 256))
    print(torch.equal(x[:, 1:], y[:, :-1]))
    print(decode(x[0].tolist()))

