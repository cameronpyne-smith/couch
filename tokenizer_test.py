import pytest

from tokenizer import Tokenizer

TEXT = "hello"


@pytest.fixture
def tokenizer():
    return Tokenizer(TEXT)


def test_decode_is_the_inverse_of_encode(tokenizer):
    assert tokenizer.decode(tokenizer.encode(TEXT)) == TEXT


def test_vocab_size_is_number_of_distinct_characters(tokenizer):
    assert tokenizer.vocab_size == 4


def test_ids_follow_sorted_character_order(tokenizer):
    assert tokenizer.encode(TEXT) == [1, 0, 2, 2, 3]


def test_unknown_character_raises(tokenizer):
    with pytest.raises(KeyError):
        tokenizer.encode("x")
