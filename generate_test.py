from generate import generate
from model import GPT, ModelConfig
from tokeniser import Tokeniser

def test_training_mode():
    model = GPT(ModelConfig(vocab_size=4, block_size=8, n_layer=1, n_head=1, n_embedding_dimension=8))
    tokeniser = Tokeniser("abc\n")
    assert model.training
    generate(model, tokeniser, "\n", 10)
    assert model.training
