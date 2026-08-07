#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARLA_ROOT="${CARLA_ROOT:-/home/chetana/opt/carla-0.9.16}"
SERVER_LOG="$PROJECT_ROOT/artifacts/evaluations/carla_server_phase4_closed_loop.log"
CHECKPOINT="${PHASE4_CHECKPOINT:-artifacts/checkpoints/phase4_single_frame_bc_v062/best.pt}"
REPORT="${PHASE4_REPORT:-artifacts/evaluations/phase4_closed_loop_report.json}"
EPISODES="${PHASE4_EPISODES:-3}"
TICKS="${PHASE4_TICKS_PER_EPISODE:-600}"
BACKGROUND="${PHASE4_BACKGROUND_VEHICLES:-8}"
RENDER_MODE="${CARLA_RENDER_MODE:-offscreen}"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "${CONDA_DEFAULT_ENV:-}" != "carla310" ]]; then
    echo "FAIL: activate carla310 before closed-loop evaluation."
    exit 1
fi
if [[ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then
    echo "FAIL: CarlaUE4.sh not found at $CARLA_ROOT"
    exit 1
fi
if [[ ! -f "$PROJECT_ROOT/$CHECKPOINT" ]]; then
    echo "FAIL: checkpoint not found at $PROJECT_ROOT/$CHECKPOINT"
    exit 1
fi

mkdir -p "$(dirname "$SERVER_LOG")"
SERVER_ARGS=(-quality-level=Low -carla-rpc-port=2000)
CLIENT_ARGS=()
if [[ "$RENDER_MODE" == "live" ]]; then
    if [[ -z "${DISPLAY:-}" ]]; then
        echo "FAIL: live rendering requires an active graphical DISPLAY."
        exit 1
    fi
    SERVER_ARGS+=(-windowed -ResX=1280 -ResY=720)
    CLIENT_ARGS+=(--spectator-follow)
    echo "Starting live CARLA closed-loop evaluation"
elif [[ "$RENDER_MODE" == "offscreen" ]]; then
    SERVER_ARGS+=(-RenderOffScreen)
    echo "Starting off-screen CARLA closed-loop evaluation"
else
    echo "FAIL: CARLA_RENDER_MODE must be 'live' or 'offscreen'."
    exit 1
fi

"$CARLA_ROOT/CarlaUE4.sh" "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID="$!"

echo "Waiting for CARLA RPC server and initial world"
python - <<'PY'
import sys
import time

import carla

for _ in range(120):
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
python scripts/evaluate_single_frame_bc_closed_loop.py \
    --checkpoint "$CHECKPOINT" \
    --report "$REPORT" \
    --episodes "$EPISODES" \
    --ticks-per-episode "$TICKS" \
    --background-vehicles "$BACKGROUND" \
    "${CLIENT_ARGS[@]}"

echo "Phase 4 closed-loop evaluation completed."
