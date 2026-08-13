#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARLA_ROOT="${CARLA_ROOT:-/home/chetana/opt/carla-0.9.16}"
SERVER_LOG="$PROJECT_ROOT/artifacts/evaluations/carla_server_phase5_closed_loop.log"
CHECKPOINT="${PHASE5_CHECKPOINT:-artifacts/checkpoints/phase5_temporal_bc_v080/best.pt}"
REPORT="${PHASE5_REPORT:-artifacts/evaluations/phase5_closed_loop_report.json}"
EPISODES="${PHASE5_EPISODES:-3}"
TICKS="${PHASE5_TICKS_PER_EPISODE:-600}"
BACKGROUND="${PHASE5_BACKGROUND_VEHICLES:-8}"
RENDER_MODE="${CARLA_RENDER_MODE:-offscreen}"
CONTINUE_AFTER_COLLISION="${PHASE5_CONTINUE_AFTER_COLLISION:-0}"
CONTROLLER="${PHASE5_CONTROLLER:-policy}"
TELEMETRY_DIR="${PHASE5_TELEMETRY_DIR:-artifacts/evaluations/phase5_telemetry}"
SAVE_VIDEO="${PHASE5_SAVE_VIDEO:-0}"
VIDEO_DIR="${PHASE5_VIDEO_DIR:-artifacts/evaluations/phase5_videos}"
EVALUATOR="${PHASE5_EVALUATOR:-scripts/evaluate_temporal_bc_closed_loop.py}"
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
            echo "CARLA did not stop gracefully; forcing owned process group shutdown"
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
    echo "FAIL: activate carla310 before closed-loop evaluation."
    exit 1
fi
if [[ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then
    echo "FAIL: CarlaUE4.sh not found at $CARLA_ROOT"
    exit 1
fi
if [[ "$CONTROLLER" == "policy" && ! -f "$PROJECT_ROOT/$CHECKPOINT" ]]; then
    echo "FAIL: checkpoint not found at $PROJECT_ROOT/$CHECKPOINT"
    exit 1
fi
if [[ "$CONTROLLER" == "expert" ]]; then
    CLIENT_ARGS+=(--expert)
    echo "Using CARLA Traffic Manager expert oracle"
elif [[ "$CONTROLLER" != "policy" ]]; then
    echo "FAIL: PHASE5_CONTROLLER must be 'policy' or 'expert'."
    exit 1
fi

mkdir -p "$(dirname "$SERVER_LOG")"
SERVER_ARGS=(-quality-level=Low -carla-rpc-port=2000)
if [[ "$RENDER_MODE" == "live" ]]; then
    if [[ -z "${DISPLAY:-}" ]]; then
        echo "FAIL: live rendering requires an active graphical DISPLAY."
        exit 1
    fi
    SERVER_ARGS+=(-windowed -ResX=1280 -ResY=720)
    CLIENT_ARGS+=(--spectator-follow)
    echo "Starting live CARLA temporal closed-loop evaluation"
elif [[ "$RENDER_MODE" == "offscreen" ]]; then
    SERVER_ARGS+=(-RenderOffScreen)
    echo "Starting off-screen CARLA temporal closed-loop evaluation"
else
    echo "FAIL: CARLA_RENDER_MODE must be 'live' or 'offscreen'."
    exit 1
fi
if [[ "$CONTINUE_AFTER_COLLISION" == "1" ]]; then
    CLIENT_ARGS+=(--continue-after-collision)
    echo "Visualization override enabled: collisions will be recorded without termination"
elif [[ "$CONTINUE_AFTER_COLLISION" != "0" ]]; then
    echo "FAIL: PHASE5_CONTINUE_AFTER_COLLISION must be 0 or 1."
    exit 1
fi
if [[ "$SAVE_VIDEO" == "1" ]]; then
    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "FAIL: PHASE5_SAVE_VIDEO=1 requires ffmpeg."
        exit 1
    fi
    CLIENT_ARGS+=(--video-dir "$VIDEO_DIR")
    echo "Saving 10 Hz ego-camera videos under $VIDEO_DIR"
elif [[ "$SAVE_VIDEO" != "0" ]]; then
    echo "FAIL: PHASE5_SAVE_VIDEO must be 0 or 1."
    exit 1
fi

python - <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    if sock.connect_ex(("127.0.0.1", 2000)) == 0:
        print("FAIL: port 2000 is already occupied; stop the existing CARLA server first.")
        sys.exit(1)
PY

setsid "$CARLA_ROOT/CarlaUE4.sh" "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"
export PHASE5_OWNED_SERVER_PID="$SERVER_PID"

echo "Waiting for CARLA RPC server and initial world"
python - <<'PY'
import os
import sys
import time

import carla

server_pid = int(os.environ["PHASE5_OWNED_SERVER_PID"])
for _ in range(120):
    try:
        os.killpg(server_pid, 0)
    except ProcessLookupError:
        print("FAIL: the newly launched CARLA server exited before RPC became ready")
        sys.exit(1)
    try:
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(3.0)
        version = client.get_server_version()
        map_name = client.get_world().get_map().name
        print(f"CARLA RPC is ready (server={version}, map={map_name})")
        break
    except RuntimeError:
        time.sleep(1)
else:
    print("FAIL: CARLA RPC did not become ready within the startup window")
    sys.exit(1)
PY

cd "$PROJECT_ROOT"
python "$EVALUATOR" \
    --checkpoint "$CHECKPOINT" \
    --report "$REPORT" \
    --episodes "$EPISODES" \
    --ticks-per-episode "$TICKS" \
    --background-vehicles "$BACKGROUND" \
    --telemetry-dir "$TELEMETRY_DIR" \
    "${CLIENT_ARGS[@]}"

echo "Phase 5 temporal closed-loop evaluation completed."
