#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

API_URL="http://localhost:8000"
WEB_URL="http://localhost:5173"
VENV_DIR=".venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed or is not on PATH."
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is not installed or is not on PATH."
  echo "Install Node.js/npm, then run ./start_locally.sh again."
  exit 1
fi

if [ ! -f .env ]; then
  touch .env
  echo "Created empty .env. Add API keys or DAGQA_CONFIG overrides if needed."
fi

mkdir -p runs

require_free_port() {
  local port="$1"
  local owner=""

  if command -v lsof >/dev/null 2>&1; then
    owner="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk 'NR==2 {print $1 " pid=" $2}' || true)"
  fi

  if [ -n "$owner" ]; then
    echo "Port $port is already in use by $owner."
    echo "Stop that process, then run ./start_locally.sh again."
    exit 1
  fi
}

if [ ! -d "$VENV_DIR" ]; then
  uv venv "$VENV_DIR"
fi

uv pip install -e ".[dev]"

if [ ! -d app/web/node_modules ]; then
  npm --prefix app/web install
fi

if docker ps --filter name=dagqa-app --filter status=running --format '{{.Names}}' 2>/dev/null | grep -qx 'dagqa-app'; then
  echo "Stopping dagqa-app Docker container to free port 8000 for local reload server."
  docker stop dagqa-app >/dev/null
fi

require_free_port 8000
require_free_port 5173

cleanup() {
  if [ -n "${API_PID:-}" ] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi
  if [ -n "${WEB_PID:-}" ] && kill -0 "$WEB_PID" 2>/dev/null; then
    kill "$WEB_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting FastAPI with hot reload at $API_URL"
"$VENV_DIR/bin/uvicorn" app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --reload-dir app \
  --reload-dir dagqa \
  --reload-dir configs \
  --reload-dir tests \
  --reload-exclude ".cache/*" \
  --reload-exclude ".venv/*" \
  --reload-exclude "app/web/node_modules/*" \
  --reload-exclude "app/web/dist/*" \
  --reload-exclude "runs/*" \
  --env-file .env &
API_PID=$!

echo "Starting Vite with hot reload at $WEB_URL"
npm --prefix app/web run dev &
WEB_PID=$!

echo
echo "Local development is running:"
echo "  Web UI:   $WEB_URL"
echo "  API docs: $API_URL/docs"
echo "  API:      $API_URL"
echo
echo "Use Ctrl+C to stop both servers."

while kill -0 "$API_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do
  sleep 1
done

wait "$API_PID" 2>/dev/null || API_STATUS=$?
wait "$WEB_PID" 2>/dev/null || WEB_STATUS=$?

exit "${API_STATUS:-${WEB_STATUS:-0}}"
