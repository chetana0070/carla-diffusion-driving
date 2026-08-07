# Phase 5: Temporal Behavioral Cloning

## Decision objective

Measure whether four-frame observation history improves offline action
prediction and closed-loop stability relative to the frozen single-frame BC
baseline. No new demonstrations, action chunks, diffusion process, safety
controller, or evaluation routes are introduced in this phase.

## Controlled architecture

| Component | Contract |
|---|---|
| Observation | Four RGB frames and four normalized 10D states |
| Visual encoder | One shared EfficientNet-B0 |
| Image projection | 1,280 to 256 per frame |
| State projection | 10 to 64 per frame |
| Temporal model | One-layer GRU, 256 hidden units |
| Condition | Current 9D route/light one-hot projected to 64 |
| Output | Current steering and longitudinal action |
| Loss | Same Phase 3 sample-weighted MSE |
| Split | Same 4,656/582/582 route-level windows |
| Selection | Lowest validation joint RMSE |

The four images in each training window receive one shared photometric
augmentation. Geometric flips remain prohibited.

## Validation sequence

```bash
python scripts/smoke_phase5_temporal_model.py

python scripts/train_temporal_bc.py \
    --smoke \
    --output-dir artifacts/checkpoints/phase5_temporal_smoke_v080
```

The architecture smoke test establishes shape, gradients, action bounds, and
VRAM demand. The training smoke test exercises all data interfaces but limits
training and evaluation to two batches.

After both pass, run a one-epoch full-route numerical preflight before the full
15-epoch experiment:

```bash
python scripts/train_temporal_bc.py \
    --pretrained \
    --epochs 1 \
    --output-dir artifacts/checkpoints/phase5_temporal_preflight_v080
```

## Exit gate

- No non-finite values or CUDA out-of-memory failures;
- all 582 validation and test windows complete in the preflight;
- full training selects a finite checkpoint using validation only;
- offline metrics are compared against the frozen epoch-5 single-frame model;
- batch-one end-to-end latency is below 100 ms;
- the checkpoint is evaluated on the identical three closed-loop seeds.

Temporal BC is successful only if it materially improves closed-loop route
progress or safety. Better offline MSE alone is insufficient.

## Closed-loop evaluation

The Phase 5 runtime uses a synchronized four-frame image/state buffer. At the
first control tick, the first observation is repeated four times so control is
available immediately. Each subsequent tick appends one observation and drops
the oldest. History is reset before every episode.

Run the model-only batch-one benchmark first:

```bash
python scripts/benchmark_temporal_bc_latency.py
```

Then run a live one-episode smoke test before the frozen three-seed evaluation:

```bash
CARLA_RENDER_MODE=live \
PHASE5_EPISODES=1 \
PHASE5_TICKS_PER_EPISODE=300 \
PHASE5_BACKGROUND_VEHICLES=2 \
PHASE5_REPORT=artifacts/evaluations/phase5_closed_loop_smoke_v081.json \
./scripts/run_phase5_closed_loop.sh
```

The final benchmark must use off-screen rendering, three episodes, 600 ticks,
eight background vehicles, and seeds beginning at 20260901—the identical Phase
4 contract.

If a live visualization closes immediately after an early collision, use the
visualization-only continuation override:

```bash
CARLA_RENDER_MODE=live \
PHASE5_CONTINUE_AFTER_COLLISION=1 \
PHASE5_EPISODES=1 \
PHASE5_TICKS_PER_EPISODE=300 \
PHASE5_BACKGROUND_VEHICLES=2 \
PHASE5_REPORT=artifacts/evaluations/phase5_visual_debug_v082.json \
./scripts/run_phase5_closed_loop.sh
```

This override is prohibited for the frozen three-seed comparison because it
changes the termination contract.
