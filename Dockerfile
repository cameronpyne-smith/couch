FROM python:3.14-slim

LABEL org.opencontainers.image.source=https://github.com/cameronpyne-smith/couch

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=True \
    UV_NO_CACHE=True

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project && rm /usr/local/bin/uv

COPY train.py model.py optimiser.py tokeniser.py generate.py ./
RUN python -m compileall -q /app

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=True \
    PYTHONDONTWRITEBYTECODE=True \
    HOME=/tmp

USER 65534:65534
ENTRYPOINT ["python", "train.py"]
CMD ["--input-data-path", "/input/shakespeare.txt", "--output-dir", "/output"]
