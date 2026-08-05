#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARLA_ROOT="${CARLA_ROOT:-/home/chetana/opt/carla-0.9.16}"
SERVER_LOG="$PROJECT_ROOT/artifacts/evaluations/carla_server_phase1.log"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "${CONDA_DEFAULT_ENV:-}" != "carla310" ]]; then
    echo "FAIL: activate carla310 before running the smoke test."
    exit 1
fi

if [[ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then
    echo "FAIL: CarlaUE4.sh not found at $CARLA_ROOT"
    echo "Run ./scripts/install_carla_0916.sh first."
    exit 1
fi

mkdir -p "$(dirname "$SERVER_LOG")"

echo "Starting CARLA 0.9.16 at Low quality in off-screen rendering mode"
"$CARLA_ROOT/CarlaUE4.sh" \
    -quality-level=Low \
    -RenderOffScreen \
    -carla-rpc-port=2000 \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

echo "Waiting for CARLA server on port 2000"
python - <<'PY'
import socket
import sys
import time

for _ in range(120):
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", 2000)) == 0:
            print("CARLA server port is ready")
            break
    time.sleep(1)
else:
    print("FAIL: CARLA did not open port 2000 within 120 seconds")
    sys.exit(1)
PY

cd "$PROJECT_ROOT"
python scripts/phase1_sensor_smoke.py \
    --ticks 1000 \
    --seed 20260803 \
    --width 640 \
    --height 360 \
    --fixed-delta 0.1

echo "Phase 1 sensor smoke test passed."

