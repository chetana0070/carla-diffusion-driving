#!/usr/bin/env bash
set -euo pipefail

export PHASE5_CHECKPOINT="${PHASE6_RESIDUAL_CHECKPOINT:-artifacts/checkpoints/phase6_residual_v130/best.pt}"
export PHASE5_REPORT="${PHASE6_RESIDUAL_REPORT:-artifacts/evaluations/phase6_residual_closed_loop.json}"
export PHASE5_EPISODES="${PHASE6_RESIDUAL_EPISODES:-3}"
export PHASE5_TICKS_PER_EPISODE="${PHASE6_RESIDUAL_TICKS_PER_EPISODE:-600}"
export PHASE5_BACKGROUND_VEHICLES="${PHASE6_RESIDUAL_BACKGROUND_VEHICLES:-8}"
export PHASE5_TELEMETRY_DIR="${PHASE6_RESIDUAL_TELEMETRY_DIR:-artifacts/evaluations/phase6_residual_telemetry}"
export PHASE5_SAVE_VIDEO="${PHASE6_RESIDUAL_SAVE_VIDEO:-0}"
export PHASE5_VIDEO_DIR="${PHASE6_RESIDUAL_VIDEO_DIR:-artifacts/evaluations/phase6_residual_videos}"
if [[ -n "${PHASE6_RESIDUAL_SEED:-}" ]]; then
    export PHASE5_EVALUATOR="scripts/evaluate_residual_seeded_closed_loop.py"
else
    export PHASE5_EVALUATOR="scripts/evaluate_residual_closed_loop.py"
fi

exec "$(dirname "$0")/run_phase5_closed_loop.sh"
