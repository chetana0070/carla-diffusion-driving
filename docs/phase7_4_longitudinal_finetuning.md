# Phase 7.4 — Longitudinal-Only Fine-Tuning

## Decision

Phase 7.3 is retained as a valid negative baseline. Its steering, latency,
action bounds, bias, behavior-balance, and smoothness gates passed. Promotion
failed only because longitudinal RMSE was 1.288 times the temporal-BC baseline
and joint RMSE was 1.253 times the baseline, above the frozen 1.25 limit.

## Change boundary

- Warm-start from `phase7_factorized_v191/best.pt`.
- Freeze the complete perception, temporal encoder, and diffusion-steering path.
- Train only the longitudinal trunk and action head.
- Use direct, unweighted MSE because the required promotion metric is RMSE.
- Emphasize the first executed action while retaining chunk and derivative terms.
- Select checkpoints exclusively on validation first-action longitudinal RMSE.
- Never use the test split for optimization or checkpoint selection.

The steering branch remains byte-for-byte represented by the Phase 7.3 state;
only longitudinal parameters are updated by the optimizer.

## Smoke test

```bash
make phase7-longitudinal-smoke
```

## Full run

```bash
python scripts/finetune_factorized_longitudinal.py \
  --initial-checkpoint artifacts/checkpoints/phase7_factorized_v191/best.pt \
  --output-dir artifacts/checkpoints/phase7_factorized_longitudinal_v200
```

After training, benchmark `best.pt` with the existing factorized latency tool
and evaluate it with the unchanged factorized offline gate. A nonzero gate exit
status blocks CARLA deployment; thresholds must not be relaxed.
