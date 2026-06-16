# ai-agent-practical-course

DAG QA is a multi-hop question answering prototype for the AI Agents practical
course. It contains a FastAPI backend, a React/Vite frontend, benchmark tooling
for HotpotQA-style evaluations, and persistent benchmark result storage under
`runs/benchmarks`.

## Requirements

- Python 3.11+
- `uv`
- Node.js 20+ and npm
- Docker Desktop with Docker Compose, if you want to run the containerized setup
- API credentials in `.env` for live LLM calls

Create your local environment file from the example:

```bash
cp .example.env .env
```

For the university chair inference service, configure at least:

```bash
CLUSTER_API_KEY=...
CLUSTER_API_BASE=http://atknoll32.air.cit.tum.de:3000/inference
CLUSTER_MODELS_URL=http://atknoll32.air.cit.tum.de:3000/models
```

The application can also load the other configs in `configs/`, but current
benchmark work should use the chair models rather than the older OpenRouter
configs.

## Start Locally

The local startup script creates or reuses `.venv`, installs Python
dependencies, installs frontend dependencies, stops the Docker app container if
it is occupying port `8000`, and starts both development servers with hot reload:

```bash
./start_locally.sh
```

After startup:

- Frontend: `http://localhost:5173`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

Stop both local servers with `Ctrl+C`.

## Start With Docker

For the Docker-backed app, run:

```bash
./start.sh
```

This builds the frontend, builds the backend image, and starts the app with
Docker Compose. The app is served at:

```text
http://localhost:8000
```

For development with a separate Vite frontend container:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up dagqa web-dev
```

Then open:

```text
http://localhost:5173
```

Stop the containers with:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```

## Tests

Run the Python test suite locally:

```bash
uv run pytest
```

Run focused tests while developing:

```bash
uv run pytest tests/unit/test_answer_postprocess.py tests/unit/test_benchmark_results_api.py
```

Build the frontend:

```bash
npm --prefix app/web run build
```

Run the Docker test profile:

```bash
docker compose --profile test run --rm test
```

## Benchmarks

Benchmarks are started through the backend so they appear in the frontend
Results tab. Start the local app first, then run:

```bash
DAGQA_API_URL=http://127.0.0.1:8000 ./scripts/start_benchmark.sh 30
```

The script asks for the LLM model, benchmark name, optional seed, and whether to
run the DAG agent, direct LLM baseline, or both. Results are written to:

```text
runs/benchmarks/
```

Saved/marked rows from the frontend are stored persistently in:

```text
runs/benchmarks/.marked/rows.json
```

## Documentation

Project reports, design notes, benchmark analyses, and research notes live in
`docs/`. Keep `README.md` as the only Markdown file at the repository root.

Useful current reports:

- `docs/cosine_improvement_research_2026-06-16.md`
- `docs/cosine_improvement_failure_analysis_consistency_v2_2026-06-16.md`
- `docs/failed_experiments_2026-06-15.md`
