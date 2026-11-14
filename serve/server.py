import argparse
import json
import math
import signal
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from generate import generate, load_model

MAX_BODY_BYTES = 1 << 20
MAX_NEW_TOKENS = 2000
DEFAULT_NEW_TOKENS = 500
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 40
REQUEST_TIMEOUT_SECONDS = 60


class RequestError(Exception):
    pass


def prompt_of(messages):
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        raise RequestError("the last message must be the user's")
    content = last.get("content")
    if isinstance(content, list):
        parts = [p.get("text") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        if not all(isinstance(t, str) for t in parts):
            raise RequestError("a text part must hold a string")
        content = "".join(parts)
    if not isinstance(content, str):
        raise RequestError("the last message's content must be a string")
    return content


def number(body, name, default, kind, low, high):
    value = body.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (kind is int and value != int(value)):
        raise RequestError(f"{name} must be {'an integer' if kind is int else 'a number'}")
    if not low <= value <= high:
        raise RequestError(f"{name} must be from {low} to {high}")
    return kind(value)


def settings_of(body):
    tokens = {"max_tokens": body.get("max_completion_tokens", body.get("max_tokens"))}
    return {
        "prompt": prompt_of(body.get("messages")),
        "max_new_tokens": number(tokens, "max_tokens", DEFAULT_NEW_TOKENS, int, 1, MAX_NEW_TOKENS),
        "temperature": number(body, "temperature", DEFAULT_TEMPERATURE, float, 0, 2),
        "top_k": number(body, "top_k", DEFAULT_TOP_K, int, 1, 1000),
    }


def known(tokeniser, text):
    return "".join(c for c in text if c in tokeniser.char_to_id_map)


class Model:
    def __init__(self, model, tokeniser, name):
        self.model = model
        self.tokeniser = tokeniser
        self.name = name
        self.lock = threading.Lock()

    def complete(self, prompt, max_new_tokens, temperature, top_k):
        start = known(self.tokeniser, prompt) or ("\n" if "\n" in self.tokeniser.char_to_id_map else self.tokeniser.chars[0])
        if temperature == 0:
            temperature, top_k = 1.0, 1
        with self.lock:
            text = generate(self.model, self.tokeniser, start, max_new_tokens, temperature, top_k)
        return text[len(start):], len(start)

    def answer(self, body):
        s = settings_of(body)
        text, prompt_tokens = self.complete(s["prompt"], s["max_new_tokens"], s["temperature"], s["top_k"])
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(text), "total_tokens": prompt_tokens + len(text)},
        }


def handler_for(served):
    class Handler(BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT_SECONDS

        def send_json(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_error_json(self, status, message):
            self.send_json(status, {"error": {"message": message, "type": "invalid_request_error"}})

        def do_GET(self):
            if self.path in ("/", "/health"):
                self.send_json(200, {"status": "ok", "model": served.name})
            elif self.path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [{"id": served.name, "object": "model", "created": 0, "owned_by": "couch"}]})
            else:
                self.send_error_json(404, "not found")

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error_json(404, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_BODY_BYTES:
                self.send_error_json(400, f"a JSON body of at most {MAX_BODY_BYTES} bytes is required")
                return
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise RequestError("the body must be a JSON object")
                if body.get("stream"):
                    raise RequestError("stream is not supported")
                self.send_json(200, served.answer(body))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_error_json(400, "the body is not JSON")
            except RequestError as e:
                self.send_error_json(400, str(e))
            except Exception as e:
                traceback.print_exc()
                self.send_json(500, {"error": {"message": f"the model failed to answer: {type(e).__name__}", "type": "server_error"}})

        def log_message(self, format, *args):
            if self.command != "GET":
                super().log_message(format, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    args = parser.parse_args()
    model, tokeniser, step = load_model(args.checkpoint, args.device)
    served = Model(model, tokeniser, f"couch-gpt-step-{step}")
    server = ThreadingHTTPServer((args.host, args.port), handler_for(served))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f"serving {served.name} ({tokeniser.vocab_size} characters) on {args.device} at {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
