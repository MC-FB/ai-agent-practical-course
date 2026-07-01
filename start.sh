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
  echo "Created empty .env. Add CLUSTER_API_KEY for chair cluster LLM calls."
fi

mkdir -p runs

verify_web_dist_fresh() {
  if [ ! -f app/web/dist/index.html ]; then
    echo "Web UI build verification failed: app/web/dist/index.html is missing."
    return 1
  fi

  local newest_source
  newest_source="$(
    find app/web \
      \( -path app/web/node_modules -o -path app/web/dist \) -prune \
      -o -type f -newer app/web/dist/index.html -print -quit
  )"

  if [ -n "$newest_source" ]; then
    echo "Web UI build verification failed: $newest_source is newer than app/web/dist/index.html."
    return 1
  fi
}

if command -v npm >/dev/null 2>&1; then
  if [ ! -d app/web/node_modules ]; then
    npm --prefix app/web install
  fi
  npm --prefix app/web run build
  if ! verify_web_dist_fresh; then
    echo "Rebuild the UI, then run ./start.sh again."
    exit 1
  fi
elif [ ! -f app/web/dist/index.html ]; then
  echo "npm is required to build the web UI, and app/web/dist is missing."
  echo "Install Node.js/npm or build the UI another way, then run ./start.sh again."
  exit 1
else
  if verify_web_dist_fresh; then
    echo "npm not found; using verified existing app/web/dist build."
  else
    echo "npm not found and app/web/dist appears stale."
    echo "Install Node.js/npm so ./start.sh can rebuild the web UI."
    exit 1
  fi
fi

"${COMPOSE[@]}" up -d --build

echo
echo "$APP_NAME is running at $APP_URL"
echo "After this first run, Docker Desktop will show a container named '$APP_NAME'."
echo "You can stop/start that container from Docker Desktop with one click."
