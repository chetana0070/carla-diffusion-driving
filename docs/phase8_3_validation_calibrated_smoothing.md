# Phase 8.3 — Validation-Calibrated Steering Smoothing

## Decision

Phase 8.2 fixed longitudinal bias and passed every accuracy, behavior, integrity,
coverage, and latency check. Promotion remained blocked only because predicted
steering chunk variation was `2.341x` the target rather than at most `1.50x`.

Phase 8.3 does not retrain the model and does not weaken that gate. It freezes a
deterministic, causal steering transformation selected exclusively on validation
data. The untouched test split remains the final promotion surface.

## Transformation contract

For steering action `s[t]`, the derived checkpoint applies:

```text
filtered[0] = s[0]
filtered[t] = filtered[t - 1] + alpha * (s[t] - filtered[t - 1])
```

The transformation:

- preserves the complete first action exactly;
- leaves every longitudinal value unchanged;
- uses only current and previous chunk values;
- remains bounded because it is a convex combination of bounded actions;
- is included in both offline evaluation and measured planner latency;
- is serialized in `deployment_transform` metadata in the derived checkpoint.

## Validation-only selection

`scripts/calibrate_phase8_vla_smoother.py` evaluates the frozen alpha grid on all
582 validation windows. It chooses the largest alpha that:

- reaches the internal `1.25x` steering-smoothness target, creating headroom
  below the unchanged public `1.50x` gate;
- preserves action zero and all longitudinal values exactly; and
- limits validation full-chunk steering RMSE degradation to 25%.

If no candidate satisfies all requirements, calibration exits with status 2 and
does not write a candidate checkpoint. Test data is never loaded during selection.

## Local execution

The Phase 8.2 checkpoint, its training report, the prepared Phase 8 dataset, and
the underlying raw images must be present.

```bash
make phase8-vla-smoother-calibrate
make phase8-vla-smoother-latency
make phase8-vla-smoother-gate
```

Expected artifacts:

- `artifacts/checkpoints/phase8_vla_smoothed_v250/best.pt`
- `artifacts/checkpoints/phase8_vla_smoothed_v250/report.json`
- `artifacts/evaluations/phase8_vla_smoothed_latency_v250.json`
- `artifacts/evaluations/phase8_vla_smoothed_gate_v250.json`

The final gate intentionally exits nonzero if any unchanged Phase 8.1 acceptance
condition fails. CARLA integration remains prohibited until that report records
`promotion_gate_passed: true`.
