FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install only what's needed for serving (sem dev, eda, monitoring, data, experiment)
RUN uv sync --no-dev --group model --group serving

# Copy source code
COPY src/ ./src/

# Copy artefatos gerados pelo DVC (devem existir localmente após dvc repro/pull)
COPY artifacts/ ./artifacts/
COPY models/ ./models/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
