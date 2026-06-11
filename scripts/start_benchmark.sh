#!/bin/sh
set -eu

API_URL="${DAGQA_API_URL:-http://127.0.0.1:8002}"
LIMIT="${1:-}"

usage() {
  echo "Usage: $0 <num_examples>"
  echo
  echo "Environment:"
  echo "  DAGQA_API_URL  Backend URL, default: http://127.0.0.1:8002"
}

case "${LIMIT}" in
  ''|*[!0-9]*)
    usage
    exit 2
    ;;
esac

if [ "${LIMIT}" -le 0 ]; then
  echo "num_examples must be greater than 0."
  exit 2
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required."
  exit 1
fi

PYTHON="${PYTHON:-}"
if [ -z "${PYTHON}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  else
    echo "python3 or .venv/bin/python is required."
    exit 1
  fi
fi

TMP_DIR="${TMPDIR:-/tmp}"
MODELS_FILE="$(mktemp "${TMP_DIR%/}/dagqa-models.XXXXXX.json")"
BODY_FILE="$(mktemp "${TMP_DIR%/}/dagqa-benchmark-body.XXXXXX.json")"
PREFLIGHT_FILE="$(mktemp "${TMP_DIR%/}/dagqa-preflight.XXXXXX.json")"
START_FILE="$(mktemp "${TMP_DIR%/}/dagqa-start.XXXXXX.json")"
trap 'rm -f "${MODELS_FILE}" "${BODY_FILE}" "${PREFLIGHT_FILE}" "${START_FILE}"' EXIT

echo "Using API: ${API_URL}"
if ! curl -fsS "${API_URL}/api/llm/models" > "${MODELS_FILE}"; then
  echo "Could not load LLM models. Is the DAGQA backend running?"
  exit 1
fi

echo
echo "LLM models:"
"${PYTHON}" -c '
import json, sys
data = json.load(open(sys.argv[1]))
models = data.get("models") or []
default = data.get("default") or {}
for index, model in enumerate(models, 1):
    marker = " *" if model.get("provider") == default.get("provider") and model.get("model") == default.get("model") else ""
    label = model.get("label") or model.get("model")
    print(f"{index:2d}. {label}{marker}")
if data.get("cluster_error"):
    print("\nCluster model catalog warning: " + str(data["cluster_error"]))
' "${MODELS_FILE}"

MODEL_COUNT="$("${PYTHON}" -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("models") or []))' "${MODELS_FILE}")"
if [ "${MODEL_COUNT}" -eq 0 ]; then
  echo "No LLM models are available."
  exit 1
fi

DEFAULT_INDEX="$("${PYTHON}" -c '
import json, sys
data=json.load(open(sys.argv[1])); models=data.get("models") or []; default=data.get("default") or {}
for i, model in enumerate(models, 1):
    if model.get("provider")==default.get("provider") and model.get("model")==default.get("model"):
        print(i); break
else:
    print(1)
' "${MODELS_FILE}")"

printf "Select model [${DEFAULT_INDEX}]: "
read -r MODEL_CHOICE
MODEL_CHOICE="${MODEL_CHOICE:-$DEFAULT_INDEX}"

MODEL_SELECTION="$("${PYTHON}" -c '
import json, sys
data=json.load(open(sys.argv[1])); choice=int(sys.argv[2]); models=data.get("models") or []
if choice < 1 or choice > len(models):
    raise SystemExit(f"Model choice must be between 1 and {len(models)}.")
model=models[choice-1]
print(model["provider"] + "\t" + model["model"] + "\t" + (model.get("label") or model["model"]))
' "${MODELS_FILE}" "${MODEL_CHOICE}")"
PROVIDER="$(printf '%s' "${MODEL_SELECTION}" | cut -f1)"
MODEL="$(printf '%s' "${MODEL_SELECTION}" | cut -f2)"
MODEL_LABEL="$(printf '%s' "${MODEL_SELECTION}" | cut -f3-)"

echo
printf "Benchmark name [terminal-%s]: " "$(date +%Y%m%d-%H%M%S)"
read -r NAME
if [ -z "${NAME}" ]; then
  NAME="terminal-$(date +%Y%m%d-%H%M%S)"
fi

printf "Seed [random]: "
read -r SEED
case "${SEED}" in
  ''|*[!0-9]*)
    if [ -n "${SEED}" ]; then
      echo "Seed must be an integer or empty."
      exit 2
    fi
    ;;
esac

echo
echo "Systems:"
echo "  1. DAG agent"
echo "  2. Direct LLM"
echo "  3. Both"
printf "Select systems [3]: "
read -r SYSTEM_CHOICE
SYSTEM_CHOICE="${SYSTEM_CHOICE:-3}"
case "${SYSTEM_CHOICE}" in
  1) SYSTEMS_JSON='["dag_agent"]' ;;
  2) SYSTEMS_JSON='["direct_llm"]' ;;
  3) SYSTEMS_JSON='["dag_agent","direct_llm"]' ;;
  *)
    echo "System choice must be 1, 2, or 3."
    exit 2
    ;;
esac

env \
  DAGQA_BENCHMARK_LIMIT="${LIMIT}" \
  DAGQA_BENCHMARK_NAME="${NAME}" \
  DAGQA_BENCHMARK_SYSTEMS="${SYSTEMS_JSON}" \
  DAGQA_BENCHMARK_PROVIDER="${PROVIDER}" \
  DAGQA_BENCHMARK_MODEL="${MODEL}" \
  DAGQA_BENCHMARK_SEED="${SEED}" \
  "${PYTHON}" -c '
import json, os, sys
seed = os.environ.get("DAGQA_BENCHMARK_SEED")
body = {
    "limit": int(os.environ["DAGQA_BENCHMARK_LIMIT"]),
    "name": os.environ["DAGQA_BENCHMARK_NAME"],
    "systems": json.loads(os.environ["DAGQA_BENCHMARK_SYSTEMS"]),
    "llm": {
        "provider": os.environ["DAGQA_BENCHMARK_PROVIDER"],
        "model": os.environ["DAGQA_BENCHMARK_MODEL"],
    },
}
if seed:
    body["seed"] = int(seed)
json.dump(body, open(sys.argv[1], "w"))
' "${BODY_FILE}"

echo
echo "Preflight:"
if ! curl -fsS \
  -H "Content-Type: application/json" \
  -d @"${BODY_FILE}" \
  "${API_URL}/api/benchmarks/hotpotqa/preflight" > "${PREFLIGHT_FILE}"; then
  echo "Preflight request failed."
  exit 1
fi

"${PYTHON}" -c '
import json, sys
data=json.load(open(sys.argv[1]))
for check in data.get("checks", []):
    icon = "OK" if check.get("ok") else "FAIL"
    print(f"  {icon:4s} {check.get('name')}: {check.get('detail')}")
if not data.get("ok"):
    raise SystemExit(1)
' "${PREFLIGHT_FILE}" || {
  echo "Preflight failed; benchmark was not started."
  exit 1
}

echo
echo "Starting benchmark:"
echo "  examples: ${LIMIT}"
echo "  name:     ${NAME}"
echo "  model:    ${MODEL_LABEL}"
echo "  systems:  ${SYSTEMS_JSON}"

if ! curl -fsS \
  -H "Content-Type: application/json" \
  -d @"${BODY_FILE}" \
  "${API_URL}/api/benchmarks/hotpotqa/live" > "${START_FILE}"; then
  echo "Start request failed."
  exit 1
fi

"${PYTHON}" -c '
import json, sys
data=json.load(open(sys.argv[1]))
print(f"Started live benchmark {data.get('run_id')}")
print(f"Progress: {data.get('completed', 0)} / {data.get('total', 0)}")
print(f"Status: {data.get('phase')} / {data.get('status')}")
' "${START_FILE}"

echo
echo "Open the FE to monitor it:"
echo "  http://127.0.0.1:5173/"
