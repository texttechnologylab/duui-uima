#!/usr/bin/env sh
set -eu

PORT="${PORT:-9714}"
GEONAMES_BACKEND_PORT="${GEONAMES_BACKEND_PORT:-9715}"
GEONAMES_BACKEND="${GEONAMES_BACKEND:-http://127.0.0.1:${GEONAMES_BACKEND_PORT}}"
LOG_LEVEL="${LOG_LEVEL:-info}"

export PORT
export GEONAMES_BACKEND
export LOG_LEVEL

backend_pid=""
proxy_pid=""

cleanup() {
  if [ -n "${proxy_pid}" ]; then
    kill "${proxy_pid}" 2>/dev/null || true
  fi
  if [ -n "${backend_pid}" ]; then
    kill "${backend_pid}" 2>/dev/null || true
  fi
}

trap cleanup INT TERM EXIT

/app/geonames-fst \
  --port "${GEONAMES_BACKEND_PORT}" \
  /app/data/geonames/ \
  --alternate /app/data/alternateNames/ &
backend_pid="$!"

python - <<'PY'
import os
import socket
import sys
import time

host = "127.0.0.1"
port = int(os.environ.get("GEONAMES_BACKEND_PORT", "9715"))

for _ in range(120):
    try:
        with socket.create_connection((host, port), timeout=1):
            sys.exit(0)
    except OSError:
        time.sleep(0.5)

print(f"GeoNames backend did not open {host}:{port}", file=sys.stderr)
sys.exit(1)
PY

python /app/resources/duui_geonames_proxy.py &
proxy_pid="$!"

wait "${proxy_pid}"
