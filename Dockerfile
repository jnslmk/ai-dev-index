FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml README.md ./
COPY src ./src
COPY channels.yaml ./channels.yaml

RUN uv pip install --no-cache .

RUN mkdir -p /data
VOLUME ["/data"]

ENV AI_DEV_INDEX_HOST=0.0.0.0 \
    AI_DEV_INDEX_PORT=8080 \
    AI_DEV_INDEX_DATA_DIR=/data \
    AI_DEV_INDEX_CHANNELS_FILE=/app/channels.yaml

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=3)" || exit 1

CMD ["ai-dev-index", "serve"]
