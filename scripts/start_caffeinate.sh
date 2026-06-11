#!/bin/sh
set -eu

LABEL="${DAGQA_CAFFEINATE_LABEL:-dagqa-caffeinate}"
HOURS="${1:-10}"

case "${HOURS}" in
  ''|*[!0-9]*)
    echo "Usage: $0 [hours]"
    echo "hours must be a positive integer; default is 10."
    exit 2
    ;;
esac

if [ "${HOURS}" -le 0 ]; then
  echo "hours must be greater than 0."
  exit 2
fi

SECONDS=$((HOURS * 3600))

if [ -z "${DAGQA_CAFFEINATE_LABEL:-}" ]; then
  /bin/launchctl remove "dagqa-caffeinate-24h" 2>/dev/null || true
fi
/bin/launchctl remove "${LABEL}" 2>/dev/null || true
/bin/launchctl submit -l "${LABEL}" -- /usr/bin/caffeinate -i -s -t "${SECONDS}"

echo "Started caffeinate for ${HOURS} hour(s)."
echo "launchctl label: ${LABEL}"
echo "Stop with: scripts/stop_caffeinate.sh"
