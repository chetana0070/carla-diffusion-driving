# Phase 8.4 — Pareto-Robust VLA Smoothing

## Decision

Phase 8.3 selected `alpha=0.5` as the largest validation-eligible coefficient.
It passed every test condition except steering chunk smoothness, which reached
`1.5176x` against the unchanged `1.50x` limit. This was a valid negative result,
not an execution failure.

Phase 8.4 does not hard-code a coefficient from that test result. It reruns the
unchanged validation alpha grid and applies a frozen accuracy/smoothness Pareto
rule. The previously observed test result is retained only as provenance.

## Selection contract

The calibrator first applies the existing Phase 8.3 eligibility requirements:

- validation steering smoothness no greater than `1.25x` target;
- validation full-chunk steering RMSE degradation no greater than 25%;
- exact preservation of action zero and every longitudinal value.

Among eligible candidates, it finds the best validation full-chunk steering
RMSE. Candidates within 0.5% of that value are treated as accuracy-equivalent.
The candidate with the lowest validation steering smoothness ratio is selected.
Ties are resolved by lower RMSE and then larger alpha.

The recorded Phase 8.3 validation table indicates that `alpha=0.4` should win:
its steering RMSE differs from the best candidate by approximately 0.15%, while
`alpha=0.3` is outside the frozen 0.5% equivalence band. The code derives this
outcome from validation data; the coefficient is not embedded in the rule.

## Test-exposure governance

The original test split was observed in Phase 8.3. Phase 8.4 therefore:

- verifies and hashes the Phase 8.3 gate report;
- records that report in checkpoint and calibration metadata;
- explicitly records that test metrics are not used for Pareto selection;
- classifies any rerun on that split as a development regression;
- reports technical gate status separately from promotion authority; and
- forces the `fresh_holdout` promotion check to fail until a new route-level
  holdout is collected and evaluated.

This prevents a successful rerun from being presented as unbiased test evidence.

## Execution

Required local inputs are the Phase 8.2 checkpoint and report, Phase 8.3 gate
report, prepared Phase 8 dataset, and underlying raw images.

```bash
make phase8-vla-pareto-calibrate
make phase8-vla-pareto-latency
make phase8-vla-pareto-development-gate
```

Expected artifacts:

- `artifacts/checkpoints/phase8_vla_pareto_v260/best.pt`
- `artifacts/checkpoints/phase8_vla_pareto_v260/report.json`
- `artifacts/evaluations/phase8_vla_pareto_latency_v260.json`
- `artifacts/evaluations/phase8_vla_pareto_development_gate_v260.json`

The development gate is expected to exit with status 2 because the fresh-holdout
requirement remains unsatisfied, even when every technical regression check passes.
Do not run CARLA from this result.
