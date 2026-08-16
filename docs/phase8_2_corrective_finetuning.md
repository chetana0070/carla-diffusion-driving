# Phase 8.2 — Corrective VLA Fine-Tuning

## Decision

Phase 8.1 remains a released negative baseline. It passed integrity, coverage,
bounded-output, RMSE, behavior, and latency checks, but it is not eligible for
CARLA deployment because first-action longitudinal bias was `+0.1073` and the
steering chunk smoothness ratio was `6.23`.

The promotion thresholds are frozen. Phase 8.2 changes the training objective
and validation selection process rather than weakening acceptance criteria.

## Change boundary

- Warm-start from `artifacts/checkpoints/phase8_vla_v220/best.pt`.
- Freeze the EfficientNet visual encoder, including BatchNorm running state.
- Fine-tune the visual projection, state encoder, language path, fusion trunk,
  and action head.
- Preserve the 4-frame observation history and 16-action output horizon.
- Use axis-specific chunk and first-action MSE terms.
- Match predicted and target adjacent-action derivatives, with stronger steering
  weighting to suppress oscillation.
- Penalize squared first-action longitudinal bias.
- Apply training-only condition weights to `straight`, `yellow`, and `green`
  examples identified by the Phase 8.1 diagnostic slices.

Validation checkpoint selection minimizes first-action joint RMSE plus normalized
penalties when longitudinal bias exceeds `0.075` or steering smoothness exceeds
`1.25x`. These internal targets create headroom beneath the unchanged public gates
of `0.10` bias and `1.50x` chunk smoothness. Test data is evaluated once, after the
best validation checkpoint is frozen.

## Local smoke test

The released Phase 8.0 checkpoint and prepared Phase 8 dataset must be present.

```bash
make phase8-vla-corrective-smoke
```

The smoke run validates checkpoint loading, the corrective objective, backward
propagation, validation selection, checkpoint writing, and final test-report
generation. It is not an accuracy result.

## SOL A100 training

Synchronize the updated repository to the existing SOL workspace and submit:

```bash
make phase8-vla-corrective-sol-submit
```

The job writes:

- `artifacts/checkpoints/phase8_vla_corrective_v240/best.pt`
- `artifacts/checkpoints/phase8_vla_corrective_v240/report.json`
- `artifacts/phase8-vla-fix-<job-id>.out`
- `artifacts/phase8-vla-fix-<job-id>.err`

## Promotion sequence

Copy the checkpoint and report back to the local GPU workstation, then run:

```bash
make phase8-vla-corrective-latency
make phase8-vla-corrective-gate
```

The second command intentionally exits nonzero when any unchanged Phase 8.1 gate
fails. Do not run CARLA, tune thresholds, or inspect test examples to guide another
training run unless the complete offline promotion contract passes.
