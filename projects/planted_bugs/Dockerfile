FROM python:3.14-slim

LABEL org.opencontainers.image.source=https://github.com/cameronpyne-smith/couch

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=bind,from=ghcr.io/astral-sh/uv:0.11.14,source=/uv,target=/usr/local/bin/uv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    uv sync --frozen --no-dev --no-install-project --no-install-package triton \
        --no-python-downloads --compile-bytecode --no-cache

ARG MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct
ARG REVISION=2e1fd397ee46e1388853d2af2c993145b0f1098a
RUN /opt/venv/bin/python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir='/opt/model', allow_patterns=['*.json', '*.safetensors', '*.txt', 'LICENSE'])" "$MODEL" "$REVISION" \
    && rm -rf /opt/model/.cache

COPY train.py model.py optimiser.py tokeniser.py generate.py ./
COPY finetune ./finetune
COPY projects/__init__.py ./projects/
COPY projects/planted_bugs ./projects/planted_bugs

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=True \
    TORCH_DISABLE_NATIVE_JIT=1 \
    HF_HUB_OFFLINE=1 \
    HOME=/tmp

USER 65534:65534
ENTRYPOINT ["python", "-m", "projects.planted_bugs.train", "--model", "/opt/model"]
CMD ["--input-data-path", "/input/planted-bugs.jsonl", "--output-dir", "/output", "--steps", "50"]
