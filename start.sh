#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

APP_NAME="dagqa-app"
APP_URL="http://localhost:8000"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or is not on PATH."
  echo "Install Docker Desktop, then run ./start.sh again."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is not running."
  echo "Start Docker Desktop, then run ./start.sh again."
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE=(docker-compose)
else
  echo "Docker Compose is not available."
  echo "Install Docker Desktop with Compose support, then run ./start.sh again."
  exit 1
fi

if [ ! -f .env ]; then
  touch .env
  echo "Created empty .env. Add GEMINI_API_KEY or choose another config if you need live LLM calls."
fi

mkdir -p runs

if command -v npm >/dev/null 2>&1; then
  if [ ! -d app/web/node_modules ]; then
    npm --prefix app/web install
  fi
  npm --prefix app/web run build
elif [ ! -f app/web/dist/index.html ]; then
  echo "npm is required to build the web UI, and app/web/dist is missing."
  echo "Install Node.js/npm or build the UI another way, then run ./start.sh again."
  exit 1
else
  echo "npm not found; using existing app/web/dist build."
fi

"${COMPOSE[@]}" up -d --build

echo
echo "$APP_NAME is running at $APP_URL"
echo "After this first run, Docker Desktop will show a container named '$APP_NAME'."
echo "You can stop/start that container from Docker Desktop with one click."
