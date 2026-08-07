#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARLA_ROOT="${CARLA_ROOT:-/home/chetana/opt/carla-0.9.16}"
SERVER_LOG="$PROJECT_ROOT/artifacts/evaluations/carla_server_phase6.log"
RENDER_MODE="${CARLA_RENDER_MODE:-offscreen}"
CHECKPOINT="${PHASE6_CHECKPOINT:-artifacts/checkpoints/phase5_temporal_bc_v080/best.pt}"
DATASET_ROOT="${PHASE6_DATASET_ROOT:-data/raw/phase6_corrections_v1}"
REPORT="${PHASE6_REPORT:-artifacts/evaluations/phase6_collection_report.json}"
EPISODES="${PHASE6_EPISODES:-10}"
TICKS="${PHASE6_TICKS_PER_EPISODE:-300}"
BACKGROUND="${PHASE6_BACKGROUND_VEHICLES:-8}"
SEED="${PHASE6_SEED:-20261001}"
SERVER_PID=""
CLIENT_ARGS=()

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 -- "-$SERVER_PID" 2>/dev/null; then
        echo "Stopping owned CARLA server process group $SERVER_PID"
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        for _ in {1..25}; do
            if ! kill -0 -- "-$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 0.2
        done
        if kill -0 -- "-$SERVER_PID" 2>/dev/null; then
            kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        fi
    fi
    if [[ -n "$SERVER_PID" ]]; then
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${CONDA_DEFAULT_ENV:-}" != "carla310" ]]; then
    echo "FAIL: activate carla310 before corrective collection."
    exit 1
fi
if [[ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then
    echo "FAIL: CarlaUE4.sh not found at $CARLA_ROOT"
    exit 1
fi
if [[ ! -f "$PROJECT_ROOT/$CHECKPOINT" ]]; then
    echo "FAIL: temporal checkpoint not found at $PROJECT_ROOT/$CHECKPOINT"
    exit 1
fi

SERVER_ARGS=(-quality-level=Low -carla-rpc-port=2000)
if [[ "$RENDER_MODE" == "live" ]]; then
    if [[ -z "${DISPLAY:-}" ]]; then
        echo "FAIL: live rendering requires an active DISPLAY."
        exit 1
    fi
    SERVER_ARGS+=(-windowed -ResX=1280 -ResY=720)
    CLIENT_ARGS+=(--spectator-follow)
    echo "Starting live Phase 6 corrective collection"
elif [[ "$RENDER_MODE" == "offscreen" ]]; then
    SERVER_ARGS+=(-RenderOffScreen)
    echo "Starting off-screen Phase 6 corrective collection"
else
    echo "FAIL: CARLA_RENDER_MODE must be live or offscreen."
    exit 1
fi

python - <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    if sock.connect_ex(("127.0.0.1", 2000)) == 0:
        print("FAIL: port 2000 is occupied; stop the existing CARLA server first.")
        sys.exit(1)
PY

mkdir -p "$(dirname "$SERVER_LOG")"
setsid "$CARLA_ROOT/CarlaUE4.sh" "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"
export PHASE6_OWNED_SERVER_PID="$SERVER_PID"

echo "Waiting for the owned CARLA server"
python - <<'PY'
import os
import sys
import time

import carla

server_pid = int(os.environ["PHASE6_OWNED_SERVER_PID"])
for _ in range(120):
    try:
        os.killpg(server_pid, 0)
    except ProcessLookupError:
        print("FAIL: the new CARLA server exited before RPC was ready")
        sys.exit(1)
    try:
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(3.0)
        print(f"CARLA RPC is ready (server={client.get_server_version()})")
        break
    except RuntimeError:
        time.sleep(1)
else:
    print("FAIL: CARLA RPC did not become ready")
    sys.exit(1)
PY

cd "$PROJECT_ROOT"
python scripts/collect_phase6_corrections.py \
    --checkpoint "$CHECKPOINT" \
    --dataset-root "$DATASET_ROOT" \
    --report "$REPORT" \
    --episodes "$EPISODES" \
    --ticks-per-episode "$TICKS" \
    --background-vehicles "$BACKGROUND" \
    --seed "$SEED" \
    "${CLIENT_ARGS[@]}"

python scripts/audit_phase6_corrections.py "$DATASET_ROOT"
echo "Phase 6 corrective collection and audit completed."
