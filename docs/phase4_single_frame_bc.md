# Phase 4: Single-Frame Behavioral Cloning

## Purpose

Establish the simplest learned driving baseline before adding temporal context
or diffusion. The model sees the current RGB frame, current normalized state,
route command, and traffic-light condition. It predicts only the current
steering and longitudinal action.

## Scientific contract

- The route-level Phase 3 train/validation/test split is reused unchanged.
- Training uses Phase 3 sample weights to reduce redundant stationary holds and
  increase rare maneuver influence.
- The validation route selects the checkpoint by joint action RMSE.
- The test route is evaluated once after checkpoint selection.
- Horizontal flips are prohibited because they invalidate route and steering
  semantics.
- Outputs are bounded to `[-1, 1]` with `tanh`.
- This model intentionally ignores the previous three images and remaining 15
  expert actions. Temporal BC and diffusion must beat this baseline fairly.

## Architecture

| Component | Contract |
|---|---|
| Image encoder | EfficientNet-B0, 1,280 features |
| Scalar context | 10 normalized state + 9 categorical features |
| Scalar encoder | 19 to 128 features |
| Fusion head | 1,408 to 256 to 128 to 2 |
| Loss | Sample-weighted two-action MSE |
| Optimizer | AdamW, learning rate 3e-4 |
| Default batch | 32 with CUDA automatic mixed precision |
| Selection | Lowest validation joint RMSE |

## Validation sequence

The model-only smoke test checks GPU forward/backward execution without touching
the dataset:

```bash
python scripts/smoke_phase4_model.py
```

Then run a bounded two-batch end-to-end training smoke test:

```bash
python scripts/train_single_frame_bc.py \
    --smoke \
    --output-dir artifacts/checkpoints/phase4_single_frame_smoke
```

After both pass, run the full pilot baseline with ImageNet initialization:

```bash
python scripts/train_single_frame_bc.py --pretrained
```

The full run writes `best.pt` and `report.json` under
`artifacts/checkpoints/phase4_single_frame_bc/`. These artifacts and the raw
dataset remain local and are ignored by Git.

## Exit gate

- CUDA model smoke test passes on the RTX 5060;
- end-to-end smoke training reads all three route splits and saves a checkpoint;
- full training completes without NaN/Inf values;
- best epoch is chosen from validation joint RMSE;
- test MAE, RMSE, and tolerance rates are recorded;
- checkpoint inference fits inside the 100 ms control budget before closed-loop
  evaluation begins.
