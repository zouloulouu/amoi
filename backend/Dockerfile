# Backend FastAPI (ina_api) container.
#
# Build context: REPO ROOT (so we can also COPY ../dictionaries.json and
# data/clean/CURRENT if mounted via volumes at runtime).
#
# Build:    docker build -f backend/Dockerfile -t ina-api .
# Run dev:  docker run --rm -p 8000:8000 -e HF_TOKEN=... -v $(pwd)/data:/app/data -v $(pwd)/dictionaries.json:/app/dictionaries.json ina-api

FROM python:3.12-slim-bookworm AS base

# Install system deps (libs for pyarrow are already in slim, but we add curl
# for the healthcheck and ca-certificates for HTTPS to HF).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reproducible dependency installs.
COPY --from=ghcr.io/astral-sh/uv:0.5.18 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# 1) Install dependencies first (cached layer until pyproject.toml changes).
COPY backend/pyproject.toml /app/backend/pyproject.toml
RUN uv pip install --system --no-cache \
    "fastapi[standard]>=0.110" \
    "pandas>=2.0" \
    "pyarrow>=15" \
    "requests>=2.30" \
    "huggingface-hub>=0.20"

# 2) Copy backend source and install ina_core editable (so prewarm.py etc
#    are picked up without a separate build step).
COPY backend/ /app/backend/
RUN uv pip install --system --no-cache -e /app/backend

# 3) Default project layout: dictionaries state + data/ are mounted as volumes.
#    See docker-compose.yml for the canonical mounts.
ENV INA_PROJECT_ROOT=/app

EXPOSE 8000

# Drop privileges. /app must stay writable because dictionaries.json is updated
# atomically via a sibling temporary file before os.replace().
RUN useradd --create-home --shell /bin/bash --uid 1000 ina \
    && chown -R ina:ina /app
USER ina

# Healthcheck: FastAPI lifespan must have completed.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health | grep -q '"status":"ok"' || exit 1

CMD ["python", "-m", "uvicorn", "ina_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
