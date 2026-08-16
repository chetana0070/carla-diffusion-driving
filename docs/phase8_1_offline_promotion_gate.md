# Phase 8.1 — Offline VLA promotion gate

## Decision objective

Evaluate the frozen Phase 8 SOL checkpoint before it can enter CARLA runtime integration.
This phase does not retrain the planner and does not modify the Phase 7 baseline.

## Evidence contract

The gate requires:

- the checkpoint digest and epoch to match the frozen training report;
- all 582 validation and 582 test windows to be accounted for;
- bounded first-action steering and longitudinal output;
- steering, longitudinal, and joint RMSE no more than 1.25 times the promoted Phase 7
  factorized-policy baseline;
- absolute longitudinal bias no greater than 0.10;
- acceleration, braking, and neutral behavior drift no greater than 0.20;
- predicted steering and longitudinal chunk-step magnitudes no more than 1.5 times target;
- a latency report produced from the same checkpoint with p95 no greater than 500 ms.

Metrics are also broken down by observed route command and traffic-light state. Those slices
are diagnostics, not evidence of open-vocabulary grounding. Instructions remain deterministic
templates derived from structured simulator metadata.

## Execution

```bash
make phase8-vla-latency
make phase8-vla-offline-gate
```

The offline evaluator exits with status 2 when evidence is valid but promotion fails. A failed
promotion must be preserved as an experimental result; thresholds must not be weakened after
viewing test results.

## Claim boundary

Passing permits implementation of a guarded CARLA preflight. It does not establish independent
learned-policy safety, open-vocabulary comprehension, or closed-loop route performance.
