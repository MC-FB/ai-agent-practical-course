# syntax=docker/dockerfile:1.7

FROM python:3.13-slim AS backend

WORKDIR /app
COPY pyproject.toml README.md ./
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -c "import tomllib; data=tomllib.load(open('pyproject.toml','rb')); print('\n'.join(data['project']['dependencies']))" > /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install torch --index-url "$PYTORCH_INDEX_URL"
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.txt
COPY dagqa ./dagqa
COPY app ./app
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- test stage: builds on backend, adds dev dependencies ---
FROM backend AS test
RUN --mount=type=cache,target=/root/.cache/pip pip install ".[dev]"
CMD ["pytest", "tests/", "-v"]
