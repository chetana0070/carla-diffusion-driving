#!/usr/bin/env bash
set -euo pipefail

export PHASE5_CHECKPOINT="${PHASE7_CHECKPOINT:-artifacts/checkpoints/phase7_factorized_longitudinal_v200/best.pt}"
export PHASE5_REPORT="${PHASE7_REPORT:-artifacts/evaluations/phase7_factorized_closed_loop_v210.json}"
export PHASE5_EPISODES="${PHASE7_EPISODES:-3}"
export PHASE5_TICKS_PER_EPISODE="${PHASE7_TICKS_PER_EPISODE:-300}"
export PHASE5_BACKGROUND_VEHICLES="${PHASE7_BACKGROUND_VEHICLES:-8}"
export PHASE5_TELEMETRY_DIR="${PHASE7_TELEMETRY_DIR:-artifacts/evaluations/phase7_factorized_closed_loop_v210_telemetry}"
export PHASE5_SAVE_VIDEO="${PHASE7_SAVE_VIDEO:-1}"
export PHASE5_VIDEO_DIR="${PHASE7_VIDEO_DIR:-artifacts/evaluations/phase7_factorized_closed_loop_v210_videos}"
export PHASE5_EVALUATION_LABEL="Phase 7.5 factorized"
export CARLA_RENDER_MODE="${CARLA_RENDER_MODE:-live}"
if [[ -n "${PHASE7_SEED:-}" ]]; then
    export PHASE5_EVALUATOR="scripts/evaluate_factorized_seeded_closed_loop.py"
else
    export PHASE5_EVALUATOR="scripts/evaluate_factorized_closed_loop.py"
fi

exec "$(dirname "$0")/run_phase5_closed_loop.sh"
