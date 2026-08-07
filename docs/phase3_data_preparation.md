# Phase 3: Training-Data Preparation

## Objective

Convert the accepted Phase 2 pilot into a deterministic, leakage-resistant
training index before introducing PyTorch model code.

## Design contract

- Split complete routes, never individual frames.
- Use an 8/1/1 train/validation/test route split for the ten-route Town01 pilot.
- Build 4-frame observation histories with 16 expert action targets.
- Preserve the raw eight-element state in Phase 2.
- Produce ten model-state features by replacing unavailable distances with zero
  and appending lead-vehicle and traffic-light availability masks.
- Encode the current route command and traffic-light state as a nine-element
  one-hot condition vector.
- Clip longitudinal acceleration to plus or minus 12 m/s² only in prepared
  model features; raw measurements remain untouched.
- Compute normalization statistics from training routes only.
- Weight stationary brake-hold anchors at 0.25 while up-weighting turning,
  active braking, and lead-context anchors.

## Outputs

```text
data/processed/phase3_pilot_v1/
├── windows.jsonl
├── splits.json
├── normalization.json
└── report.json
```

`windows.jsonl` is an index. It references the Phase 2 JPEGs and does not copy
images, keeping storage growth small.

## Run

```bash
conda activate carla310
cd /home/chetana/carla-diffusion-driving
python scripts/prepare_phase3_data.py \
    data/raw/phase2_pilot_v2_v043 \
    --output-root data/processed/phase3_pilot_v1
```

## Exit gate

- 5,820 total windows;
- 8 train routes, 1 validation route, and 1 test route;
- no route identity occurs in more than one split;
- state history shape 4 × 10;
- action target shape 16 × 2;
- normalization statistics use only the 4,800 training-route samples;
- the generated index has a recorded SHA-256 digest;
- all tests, Ruff, mypy, and configuration validation pass.
