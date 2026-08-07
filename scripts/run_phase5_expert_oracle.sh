#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PHASE5_CONTROLLER=expert
export PHASE5_CONTINUE_AFTER_COLLISION=0
export PHASE5_REPORT="${PHASE5_EXPERT_REPORT:-artifacts/evaluations/phase5_expert_oracle.json}"
export PHASE5_TELEMETRY_DIR="${PHASE5_EXPERT_TELEMETRY_DIR:-artifacts/evaluations/phase5_expert_telemetry}"

exec "$PROJECT_ROOT/scripts/run_phase5_closed_loop.sh"
