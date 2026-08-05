#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="carla310"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "FAIL: conda is not available. Open a shell initialized by Miniconda."
    exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    conda create --name "$ENV_NAME" python=3.10 pip --yes
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
python -m pip install --editable "$PROJECT_ROOT[dev]"

cd "$PROJECT_ROOT"
python -m unittest discover -s tests -v
python scripts/validate_config.py
python scripts/validate_system.py --json-out artifacts/system_report.json

echo "Phase 0 bootstrap complete in Conda environment: $ENV_NAME"

