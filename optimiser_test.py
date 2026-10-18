import copy

import torch

from model import GPT, ModelConfig, cross_entropy
from optimiser import AdamW
from train import get_batch, load_data


def test_adamw_matches_torch_adamw():
    torch.manual_seed(0)
    training, _, tokeniser = load_data()
    config = ModelConfig(
        vocab_size=tokeniser.vocab_size,
        n_layer=2,
        n_head=2,
        n_embedding_dimension=64,
        block_size=64,
    )
    model_1 = GPT(config)
    model_2 = copy.deepcopy(model_1)

    optimiser_1 = AdamW(
        model_1.parameters(),
        learning_rate=0.001,
        betas=(0.9, 0.999),
        epsilon=1e-8,
        weight_decay=0.01,
    )
    optimiser_2 = torch.optim.AdamW(
        model_2.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )

    for _ in range(20):
        x, y = get_batch(training, config.block_size, 32, "cpu")

        for model, optimiser in ((model_1, optimiser_1), (model_2, optimiser_2)):
            loss = cross_entropy(model(x), y)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

    for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-5)
