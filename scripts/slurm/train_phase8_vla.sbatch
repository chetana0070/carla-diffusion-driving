#!/usr/bin/env bash
#SBATCH --job-name=phase8-vla
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --time=04:00:00
#SBATCH -G a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=artifacts/phase8-vla-%j.out
#SBATCH --error=artifacts/phase8-vla-%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PHASE8_PROJECT_ROOT:-/scratch/cchakrap/carla-diffusion-driving-v2.0.0/repository}"
ENVIRONMENT_NAME="${PHASE8_ENVIRONMENT_NAME:-carla310}"
DATASET_ROOT="${PHASE8_DATASET_ROOT:-$PROJECT_ROOT/data/raw/phase2_pilot_v2_v043}"

cd "$PROJECT_ROOT"
mkdir -p artifacts/logs artifacts/checkpoints/phase8_vla_v220

module purge
module load mamba/latest
source activate "$ENVIRONMENT_NAME"

python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable inside the allocated job"
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("PyTorch:", torch.__version__, flush=True)
PY

python scripts/train_phase8_vla.py \
    --processed-root data/processed/phase8_vla_v1 \
    --dataset-root "$DATASET_ROOT" \
    --output-dir artifacts/checkpoints/phase8_vla_v220 \
    --pretrained-visual-encoder
