#!/bin/sh
set -eu

LABEL="${DAGQA_CAFFEINATE_LABEL:-dagqa-caffeinate}"
STOPPED=0

if /bin/launchctl remove "${LABEL}" 2>/dev/null; then
  echo "Stopped caffeinate launchctl job: ${LABEL}."
  STOPPED=1
fi

if [ -z "${DAGQA_CAFFEINATE_LABEL:-}" ] && /bin/launchctl remove "dagqa-caffeinate-24h" 2>/dev/null; then
  echo "Stopped caffeinate launchctl job: dagqa-caffeinate-24h."
  STOPPED=1
fi

if [ "${STOPPED}" -eq 0 ]; then
  echo "No caffeinate launchctl job found for label: ${LABEL}."
else
  exit 0
fi
