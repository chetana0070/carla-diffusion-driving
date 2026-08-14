# Phase 7.3 — Factorized temporal policy

## Decision

Phase 7.2 is retained as a negative full-diffusion result. It matched temporal BC steering
performance but produced 3.36 times the temporal BC longitudinal RMSE and a positive
longitudinal bias. Phase 7.3 therefore factorizes the action model instead of changing the
test gate or repeatedly increasing longitudinal reconstruction weight.

## Architecture

- Shared EfficientNet-B0, state projection, GRU, and categorical conditioning.
- The released two-axis diffusion branch is warm-started exactly; only its steering output is
  deployed.
- A deterministic head predicts all 16 longitudinal actions from the shared observation context.
- A three-class auxiliary head predicts braking, neutral, and acceleration modes at each horizon
  step. Mode-balanced weights prevent the majority braking class from dominating training.
- The final action chunk stacks diffusion steering with deterministic longitudinal predictions.

This checkpoint must be described as `factorized_temporal_policy`. It is not evidence that
full-action diffusion outperformed behavioral cloning.

## Training contract

Run a small pipeline smoke test first:

```bash
python scripts/train_factorized_policy.py \
    --smoke \
    --output-dir artifacts/checkpoints/phase7_factorized_smoke_v190
```

Then warm-start from the frozen Phase 7.2 checkpoint:

```bash
python scripts/train_factorized_policy.py \
    --initial-diffusion-checkpoint \
        artifacts/checkpoints/phase7_diffusion_corrected_v180/best.pt \
    --output-dir artifacts/checkpoints/phase7_factorized_v190
```

Checkpoint selection uses sampled first-action joint RMSE on validation only. The test split is
evaluated once after selection.

## Promotion contract

Create the latency report:

```bash
python scripts/benchmark_factorized_policy_latency.py \
    --checkpoint artifacts/checkpoints/phase7_factorized_v190/best.pt \
    > artifacts/evaluations/phase7_factorized_latency_v190.json
```

Run the frozen full offline gate:

```bash
python scripts/evaluate_factorized_offline.py \
    --factorized-checkpoint artifacts/checkpoints/phase7_factorized_v190/best.pt \
    --baseline-checkpoint artifacts/checkpoints/phase5_temporal_bc_v080/best.pt \
    --latency-report artifacts/evaluations/phase7_factorized_latency_v190.json \
    --output artifacts/evaluations/phase7_factorized_offline_gate_v190.json
```

Promotion requires:

- Complete validation and untouched-test coverage.
- Steering, longitudinal, and joint RMSE no more than 1.25 times temporal BC.
- Absolute longitudinal bias no greater than 0.10.
- Acceleration, braking, and neutral fraction drift no greater than 0.20.
- Mean longitudinal step size no more than 1.5 times the target chunk value.
- Bounded actions and p95 end-to-end policy latency within 100 ms.

Do not run CARLA closed-loop evaluation unless every offline gate passes. A failed run is frozen
as an ablation result; thresholds are not changed after observing test metrics.
