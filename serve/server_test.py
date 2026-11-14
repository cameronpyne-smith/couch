import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import torch

from model import GPT, ModelConfig
from serve.server import Model, RequestError, handler_for, load_model, settings_of
from tokeniser import Tokeniser
from train import save_checkpoint


def tiny(tmp_path):
    config = ModelConfig(vocab_size=4, block_size=8, n_layer=1, n_head=1, n_embedding_dimension=8)
    tokeniser = Tokeniser("abc\n")
    save_checkpoint(GPT(config), config, tokeniser, 7, tmp_path)
    model, tokeniser, step = load_model(tmp_path / "checkpoint.pt", "cpu")
    return Model(model, tokeniser, f"couch-gpt-step-{step}")


@pytest.fixture
def server(tmp_path):
    served = tiny(tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(served))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def call(url, body=None):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    request = urllib.request.Request(url, data=data, method="GET" if data is None else "POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_load_model_reads_a_training_checkpoint(tmp_path):
    served = tiny(tmp_path)
    assert served.name == "couch-gpt-step-7"
    assert served.tokeniser.chars == ["\n", "a", "b", "c"]
    assert not served.model.training


def test_settings_take_the_last_user_message_and_defaults():
    s = settings_of({"messages": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "x"}, {"role": "user", "content": "abc"}]})
    assert s == {"prompt": "abc", "max_new_tokens": 500, "temperature": 0.8, "top_k": 40}


def test_settings_read_text_parts_and_max_completion_tokens():
    s = settings_of({"messages": [{"role": "user", "content": [{"type": "text", "text": "ab"}, {"type": "text", "text": "c"}]}], "max_completion_tokens": 9, "temperature": 0, "top_k": 3})
    assert (s["prompt"], s["max_new_tokens"], s["temperature"], s["top_k"]) == ("abc", 9, 0.0, 3)


@pytest.mark.parametrize("body", [
    {},
    {"messages": []},
    {"messages": [{"role": "assistant", "content": "a"}]},
    {"messages": [{"role": "user", "content": 3}]},
    {"messages": [{"role": "user", "content": "a"}], "max_tokens": 0},
    {"messages": [{"role": "user", "content": "a"}], "max_tokens": 2001},
    {"messages": [{"role": "user", "content": "a"}], "max_tokens": 1.5},
    {"messages": [{"role": "user", "content": "a"}], "temperature": 2.5},
    {"messages": [{"role": "user", "content": "a"}], "temperature": "hot"},
    {"messages": [{"role": "user", "content": "a"}], "top_k": 0},
    {"messages": [{"role": "user", "content": "a"}], "top_k": True},
    {"messages": [{"role": "user", "content": "a"}], "max_tokens": float("inf")},
    {"messages": [{"role": "user", "content": "a"}], "max_tokens": float("nan")},
    {"messages": [{"role": "user", "content": "a"}], "top_k": float("inf")},
    {"messages": [{"role": "user", "content": "a"}], "temperature": float("nan")},
])
def test_settings_refuse_what_is_out_of_range(body):
    with pytest.raises(RequestError):
        settings_of(body)


def test_complete_answers_only_the_continuation(tmp_path):
    served = tiny(tmp_path)
    text, prompt_tokens = served.complete("ab", 12, 0.8, 40)
    assert len(text) == 12 and prompt_tokens == 2
    assert set(text) <= set("abc\n")


def test_complete_drops_characters_the_model_never_saw(tmp_path):
    served = tiny(tmp_path)
    text, prompt_tokens = served.complete("a…Z", 5, 0.8, 40)
    assert prompt_tokens == 1 and len(text) == 5


def test_complete_starts_from_a_newline_when_nothing_is_known(tmp_path):
    served = tiny(tmp_path)
    text, prompt_tokens = served.complete("xyz", 5, 0.8, 40)
    assert prompt_tokens == 1 and len(text) == 5


def test_temperature_zero_is_greedy(tmp_path):
    served = tiny(tmp_path)
    torch.manual_seed(1)
    first, _ = served.complete("ab", 20, 0, 40)
    torch.manual_seed(2)
    second, _ = served.complete("ab", 20, 0, 40)
    assert first == second


def test_chat_completions_answer_in_openai_shape(server):
    status, body = call(f"{server}/v1/chat/completions", {"messages": [{"role": "user", "content": "abc"}], "max_tokens": 10})
    assert status == 200
    assert body["object"] == "chat.completion" and body["model"] == "couch-gpt-step-7"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant" and len(choice["message"]["content"]) == 10
    assert choice["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 10, "total_tokens": 13}


def test_root_and_models_answer(server):
    assert call(f"{server}/") == (200, {"status": "ok", "model": "couch-gpt-step-7"})
    assert call(f"{server}/health") == (200, {"status": "ok", "model": "couch-gpt-step-7"})
    status, body = call(f"{server}/v1/models")
    assert status == 200 and body["data"][0]["id"] == "couch-gpt-step-7"


def test_bad_requests_are_refused_as_errors(server):
    for body in [b"not json", b"[1]", {"messages": [{"role": "user", "content": "a"}], "stream": True}, {"messages": []}]:
        status, answer = call(f"{server}/v1/chat/completions", body)
        assert status == 400 and answer["error"]["type"] == "invalid_request_error"
    assert call(f"{server}/v1/completions", {"prompt": "a"})[0] == 404
    assert call(f"{server}/nope")[0] == 404


def test_infinite_numbers_in_the_body_are_refused_as_errors(server):
    status, answer = call(f"{server}/v1/chat/completions", b'{"messages": [{"role": "user", "content": "a"}], "max_tokens": Infinity}')
    assert status == 400 and answer["error"]["type"] == "invalid_request_error"


def test_a_failure_while_generating_is_answered_as_a_server_error(tmp_path):
    served = tiny(tmp_path)

    def fail(*_):
        raise RuntimeError("out of memory")

    served.complete = fail
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(served))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    status, answer = call(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions", {"messages": [{"role": "user", "content": "a"}]})
    httpd.shutdown()
    assert status == 500 and answer["error"] == {"message": "the model failed to answer: RuntimeError", "type": "server_error"}
