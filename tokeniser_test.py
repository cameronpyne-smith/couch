import pytest

from tokeniser import Tokeniser

TEXT = "hello"


@pytest.fixture
def tokeniser():
    return Tokeniser(TEXT)


def test_decode_is_the_inverse_of_encode(tokeniser):
    assert tokeniser.decode(tokeniser.encode(TEXT)) == TEXT


def test_vocab_size_is_number_of_distinct_characters(tokeniser):
    assert tokeniser.vocab_size == 4


def test_ids_follow_sorted_character_order(tokeniser):
    assert tokeniser.encode(TEXT) == [1, 0, 2, 2, 3]


def test_unknown_character_raises(tokeniser):
    with pytest.raises(KeyError):
        tokeniser.encode("x")
