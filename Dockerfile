FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app

# Install uv for fast, lockfile-reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY channels.yaml ./channels.yaml

# Install the locked dependency set (without dev tools) into /app/.venv.
RUN uv sync --frozen --no-dev \
    && mkdir -p /data \
    && chmod 0777 /data

VOLUME ["/data"]

ENV PATH="/app/.venv/bin:$PATH" \
    AI_DEV_INDEX_HOST=0.0.0.0 \
    AI_DEV_INDEX_PORT=8080 \
    AI_DEV_INDEX_DATA_DIR=/data \
    AI_DEV_INDEX_CHANNELS_FILE=/app/channels.yaml

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=3); assert r.status == 200" || exit 1

CMD ["ai-dev-index", "serve"]
