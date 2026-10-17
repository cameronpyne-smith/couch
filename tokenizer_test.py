from tokenizer import decode, encode, file_content

def test_decode_is_the_inverse_of_encode():
    assert decode(encode(file_content)) == file_content
