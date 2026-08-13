# Phase 7.1 offline diffusion promotion gate

The diffusion checkpoint is not allowed into CARLA because a small 16-window sampling
diagnostic cannot establish policy quality. Phase 7.1 evaluates every validation and test
window and compares first-action performance directly with the frozen temporal BC baseline.

## Leakage-resistant protocol

1. Evaluate four disclosed initial-noise seeds on all 582 validation windows.
2. Select the seed with the lowest validation first-action joint RMSE.
3. Evaluate that seed once on all 582 untouched test windows.
4. Evaluate temporal BC on the identical test observations and targets.
5. Report full 16-action chunk accuracy, smoothness, longitudinal behavior, bounds, and
   checkpoint-specific latency evidence.

Candidate selection never uses the test split. Noise MSE is retained as a training metric but
is not an offline promotion criterion.

## Promotion gates

- Complete validation and test coverage.
- No predicted action outside `[-1, 1]`.
- Steering, longitudinal, and joint RMSE each no more than 1.25 times temporal BC.
- Absolute longitudinal bias no greater than 0.10.
- Acceleration, braking, and neutral fraction drift no greater than 0.20.
- The saved latency report must reference the same checkpoint and pass the 100 ms budget.

Failure is an expected diagnostic outcome and blocks CARLA integration without invalidating
the diffusion training infrastructure.

## Command

```bash
python scripts/evaluate_diffusion_offline.py \
  --output artifacts/evaluations/phase7_diffusion_offline_gate_v170.json
```
