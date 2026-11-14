FROM python:3.14-slim

LABEL org.opencontainers.image.source=https://github.com/cameronpyne-smith/couch

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=bind,from=ghcr.io/astral-sh/uv:0.11.14,source=/uv,target=/usr/local/bin/uv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    uv sync --frozen --only-group serve --no-install-project --no-install-package triton \
        --no-python-downloads --compile-bytecode --no-cache

COPY model.py tokeniser.py generate.py ./
COPY serve/__init__.py serve/server.py ./serve/

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=True \
    TORCH_DISABLE_NATIVE_JIT=1 \
    HOME=/tmp

USER 65534:65534
EXPOSE 8080
ENTRYPOINT ["python", "-m", "serve.server", "--host", "0.0.0.0"]
