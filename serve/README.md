# serve

Loads a `checkpoint.pt` that `train.py` saved and answers prompts over HTTP, in the shape of OpenAI's chat completions, so
Windlet's Playground (or any OpenAI client) can talk to it.

```
python -m serve.server --checkpoint runs/<run>/checkpoint.pt --port 8080
```

It listens on `127.0.0.1` unless `--host` says otherwise; the image passes `--host 0.0.0.0`. It has no authentication, so
whatever runs it decides who can reach it.

- `POST /v1/chat/completions` with `messages`, `max_tokens` (or `max_completion_tokens`, 1 to 2000, default 500),
  `temperature` (0 to 2, default 0.8; 0 always picks the likeliest character) and `top_k` (default 40). The model is a
  base model, not a chat model: it continues the last user message and sees nothing before it. Characters the training
  text never held are dropped from the prompt. The answer is the continuation alone. No streaming.
- `GET /` and `GET /health` answer once the model is loaded; `GET /v1/models` names it `couch-gpt-step-<step>`.
- A request the server cannot read is answered 400, a failure while generating 500, both as `{"error": {"message", "type"}}`.

## Image

`serve/Dockerfile` installs only the `serve` dependency group (torch and numpy), so it leaves out what training and
fine-tuning need. `scripts/release.sh` builds and pushes it beside the training image under the same version, since it
reads the checkpoint that version of `train.py` writes.

```
docker build -f serve/Dockerfile -t ghcr.io/cameronpyne-smith/couch-server:<version> .
docker run --rm --gpus all -p 127.0.0.1:8080:8080 -v $PWD/runs/<run>:/input/run:ro \
  ghcr.io/cameronpyne-smith/couch-server:<version> --checkpoint /input/run/checkpoint.pt
```
