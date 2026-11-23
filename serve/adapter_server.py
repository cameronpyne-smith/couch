import argparse
import hashlib
import json
import math
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoTokenizer

from finetune.lora import load_lora_adapter
from finetune.sft import load_base, load_model

MAX_BODY_BYTES = 1 << 22
MAX_NEW_TOKENS = 16_384
THINK_END = "</think>"
ROLES = ("system", "user", "assistant")
REQUEST_TIMEOUT_SECONDS = 60


class RequestError(Exception):
    pass


class UnknownModel(Exception):
    pass


def tag_name(name):
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def number(options, name, default, kind, low, high):
    value = options.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (kind is int and value != int(value)):
        raise RequestError(f"{name} must be {'an integer' if kind is int else 'a number'}")
    if not low <= value <= high:
        raise RequestError(f"{name} must be from {low} to {high}")
    return kind(value)


def messages_of(body):
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ROLES or not isinstance(message.get("content"), str):
            raise RequestError(f"every message needs a role from {', '.join(ROLES)} and a string content")
    if messages[-1]["role"] != "user":
        raise RequestError("the last message must be the user's")
    return [{"role": message["role"], "content": message["content"]} for message in messages]


def settings_of(body):
    if body.get("stream") is not False:
        raise RequestError("stream must be false")
    think = body.get("think", False)
    if not isinstance(think, bool):
        raise RequestError("think must be true or false")
    options = body.get("options") or {}
    if not isinstance(options, dict):
        raise RequestError("options must be an object")
    return {
        "messages": messages_of(body),
        "think": think,
        "temperature": number(options, "temperature", 1.0, float, 0, 2),
        "top_k": number(options, "top_k", 20, int, 1, 1000),
        "top_p": number(options, "top_p", 0.95, float, 0, 1),
        "seed": number(options, "seed", 0, int, 0, 2**32 - 1),
        "max_new_tokens": number(options, "num_predict", MAX_NEW_TOKENS, int, 1, MAX_NEW_TOKENS),
    }


def split_thinking(text, think):
    if not think or THINK_END not in text:
        return text, None
    thinking, content = text.split(THINK_END, 1)
    return content.lstrip("\n"), thinking.strip()


def reply(name, text, think, prompt_tokens, output_tokens, stopped, seconds):
    content, thinking = split_thinking(text, think)
    message = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    return {
        "model": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "done": True,
        "done_reason": "stop" if stopped else "length",
        "total_duration": int(seconds * 1e9),
        "prompt_eval_count": prompt_tokens,
        "eval_count": output_tokens,
    }


def digest_of(model, bits, adapter):
    sha = hashlib.sha256(f"{model} {bits} bits".encode())
    if adapter:
        with open(adapter, "rb") as file:
            while block := file.read(1 << 20):
                sha.update(block)
    return "sha256:" + sha.hexdigest()


def stop_ids(model, tokeniser):
    configured = model.generation_config.eos_token_id
    configured = configured if isinstance(configured, list) else [configured]
    return {token for token in configured + [tokeniser.eos_token_id] if token is not None}


class Served:
    def __init__(self, model, tokeniser, name, digest):
        self.model = model
        self.tokeniser = tokeniser
        self.name = tag_name(name)
        self.digest = digest
        self.stops = stop_ids(model, tokeniser)
        self.lock = threading.Lock()

    def chat(self, settings):
        prompt = self.tokeniser.apply_chat_template(settings["messages"], add_generation_prompt=True, tokenize=True, return_dict=False, enable_thinking=settings["think"])
        input_ids = torch.tensor([prompt], device=self.model.device)
        if settings["temperature"] > 0:
            sampling = {"do_sample": True, "temperature": settings["temperature"], "top_k": settings["top_k"], "top_p": settings["top_p"]}
        else:
            sampling = {"do_sample": False}
        with self.lock, torch.no_grad():
            started = time.time()
            torch.manual_seed(settings["seed"])
            output = self.model.generate(input_ids, max_new_tokens=settings["max_new_tokens"], pad_token_id=self.tokeniser.pad_token_id, eos_token_id=sorted(self.stops), **sampling)
            seconds = time.time() - started
        generated = output[0, len(prompt) :].tolist()
        stopped = bool(generated) and generated[-1] in self.stops
        text = self.tokeniser.decode(generated, skip_special_tokens=True)
        return reply(self.name, text, settings["think"], len(prompt), len(generated), stopped, seconds)

    def answer(self, body):
        if tag_name(str(body.get("model"))) != self.name:
            raise UnknownModel(f"model '{body.get('model')}' not found, this server holds {self.name}")
        return self.chat(settings_of(body))


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

        def do_GET(self):
            path = self.path.rstrip("/")
            if path == "/api/tags":
                self.send_json(200, {"models": [{"name": served.name, "model": served.name, "digest": served.digest, "details": {"runner": "couch"}}]})
            elif path == "/api/version":
                self.send_json(200, {"version": "couch-adapter-server"})
            elif path == "":
                self.send_json(200, {"status": "ok", "model": served.name})
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/api/chat":
                self.send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length <= 0 or length > MAX_BODY_BYTES:
                self.send_json(400, {"error": f"a JSON body of at most {MAX_BODY_BYTES} bytes is required"})
                return
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise RequestError("the body must be a JSON object")
                self.send_json(200, served.answer(body))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.send_json(400, {"error": "the body is not JSON"})
            except RequestError as e:
                self.send_json(400, {"error": str(e)})
            except UnknownModel as e:
                self.send_json(404, {"error": str(e)})
            except Exception as e:
                traceback.print_exc()
                self.send_json(500, {"error": f"the model failed to answer: {type(e).__name__}"})

        def log_message(self, format, *args):
            if self.command != "GET":
                super().log_message(format, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face name of the base model")
    parser.add_argument("--bits", type=int, default=4, choices=(16, 4))
    parser.add_argument("--adapter", type=Path, help="adapter.pt written by finetune.sft; without it the base model answers")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--name", required=True, help="the model name clients must ask for, such as qwen3.8-27b-synion:epoch-3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=11435, type=int)
    args = parser.parse_args()

    tokeniser = AutoTokenizer.from_pretrained(args.model)
    if args.adapter:
        model, _ = load_model(args.rank, args.alpha, "cuda", args.model, args.bits)
        load_lora_adapter(model, args.adapter)
    else:
        model = load_base(args.model, args.bits)
    model.eval()
    served = Served(model, tokeniser, args.name, digest_of(args.model, args.bits, args.adapter))
    server = ThreadingHTTPServer((args.host, args.port), handler_for(served))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f"serving {served.name} ({served.digest[:19]}) at {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
