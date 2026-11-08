import json
import urllib.request

CHAT = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.8:27b"


def ask(system, user, schema, model=DEFAULT_MODEL, num_ctx=None, timeout=600):
    options = {"temperature": 0}
    if num_ctx:
        options["num_ctx"] = num_ctx
    body = {
        "model": model,
        "stream": False,
        "format": schema,
        "options": options,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    request = urllib.request.Request(CHAT, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.load(response)
    return json.loads(reply["message"]["content"])


def boolean_schema(name):
    return {"type": "object", "properties": {name: {"type": "boolean"}}, "required": [name]}
