import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from serve.adapter_server import RequestError, UnknownModel, handler_for, reply, settings_of, split_thinking, tag_name

BODY = {
    "model": "tuned:1",
    "stream": False,
    "think": False,
    "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    "options": {"temperature": 0.7, "seed": 3, "num_ctx": 24576, "num_predict": 100},
}


class Stub:
    name = "tuned:1"
    digest = "sha256:abc"

    def answer(self, body):
        if tag_name(str(body.get("model"))) != self.name:
            raise UnknownModel("model not found")
        settings = settings_of(body)
        return reply(self.name, "answer", settings["think"], 5, 2, True, 1.5)


@pytest.fixture
def address():
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Stub()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def post(address, body):
    request = urllib.request.Request(address + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def test_tag_name_adds_latest_only_without_a_tag():
    assert tag_name("tuned") == "tuned:latest"
    assert tag_name("tuned:epoch-3") == "tuned:epoch-3"
    assert tag_name("host:5000/tuned") == "host:5000/tuned:latest"


def test_settings_read_the_options_the_bench_sends():
    settings = settings_of(BODY)
    assert settings["messages"] == BODY["messages"]
    assert (settings["temperature"], settings["seed"], settings["max_new_tokens"], settings["think"]) == (0.7, 3, 100, False)
    assert (settings["top_k"], settings["top_p"]) == (20, 0.95)


def test_settings_refuse_streaming_and_bad_messages():
    with pytest.raises(RequestError):
        settings_of({**BODY, "stream": True})
    with pytest.raises(RequestError):
        settings_of({key: value for key, value in BODY.items() if key != "stream"})
    with pytest.raises(RequestError):
        settings_of({**BODY, "messages": [{"role": "assistant", "content": "a"}]})
    with pytest.raises(RequestError):
        settings_of({**BODY, "options": {"temperature": "hot"}})


def test_split_thinking_only_when_thinking_was_asked_for():
    assert split_thinking("plan</think>\n\nanswer", True) == ("answer", "plan")
    assert split_thinking("plan</think>\n\nanswer", False) == ("plan</think>\n\nanswer", None)
    assert split_thinking("answer", True) == ("answer", None)


def test_reply_has_the_fields_the_bench_reads():
    result = reply("tuned:1", "answer", False, 5, 2, False, 1.5)
    assert result["message"] == {"role": "assistant", "content": "answer"}
    assert (result["prompt_eval_count"], result["eval_count"], result["total_duration"], result["done_reason"]) == (5, 2, 1_500_000_000, "length")


def test_chat_round_trip(address):
    result = post(address, BODY)
    assert result["message"]["content"] == "answer"
    assert result["done_reason"] == "stop"


def test_tags_and_version(address):
    with urllib.request.urlopen(address + "/api/tags") as response:
        tags = json.loads(response.read())
    assert tags["models"] == [{"name": "tuned:1", "model": "tuned:1", "digest": "sha256:abc", "details": {"runner": "couch"}}]
    with urllib.request.urlopen(address + "/api/version") as response:
        assert json.loads(response.read())["version"]


def test_wrong_model_is_not_found_and_bad_request_is_refused(address):
    with pytest.raises(urllib.error.HTTPError) as wrong:
        post(address, {**BODY, "model": "qwen3.8:27b"})
    assert wrong.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as bad:
        post(address, {**BODY, "stream": True})
    assert bad.value.code == 400
    assert "stream" in json.loads(bad.value.read())["error"]
