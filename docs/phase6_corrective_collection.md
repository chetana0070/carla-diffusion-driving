# Phase 6: Corrective Demonstration Collection

## Objective

Collect expert recovery actions only from states reached by the failed temporal
BC policy. This addresses covariate shift without duplicating routine expert
driving or contaminating the frozen validation and test routes.

## Intervention contract

The learned policy controls the vehicle until an observable precursor fires:

- absolute lane offset at least 0.45 m;
- absolute heading error at least 10 degrees;
- positive longitudinal command within 8 m of a lead vehicle;
- speed below 0.5 m/s for 20 ticks outside a red-light stop.

After intervention, CARLA Traffic Manager controls the same vehicle along the
same generated route. The collector stores 80 consecutive expert samples. All
published samples carry `events.intervention=true`; policy-controlled samples
are not mislabeled as expert demonstrations.

Each published episode must contain at least 20 expert samples, which supports
one 4-frame/16-action temporal window. Atomic publication, schema validation,
disk gates, deterministic seeds, process-group cleanup, and route-level IDs are
retained from earlier phases.

Corrective collection has a dedicated 90 GiB start gate and 85 GiB emergency
stop gate. Its bounded 800-sample full target is far smaller than the Phase 2
pilot, while the separate thresholds retain substantial simulator, checkpoint,
and training workspace on the 240 GB Ubuntu partition.

## Smoke test

```bash
CARLA_RENDER_MODE=live \
PHASE6_EPISODES=1 \
PHASE6_TICKS_PER_EPISODE=150 \
PHASE6_BACKGROUND_VEHICLES=2 \
PHASE6_SEED=20260901 \
PHASE6_DATASET_ROOT=data/raw/phase6_smoke_v100 \
PHASE6_REPORT=artifacts/evaluations/phase6_smoke_v100.json \
./scripts/run_phase6_corrections.sh
```

The smoke test passes when at least one intervention episode is published and
the structural audit passes. The full collection gate requires at least three
published intervention episodes, no collision samples, and no schema issues.

## Governance

This phase does not retrain a policy. After collection, inspect trigger balance,
action coverage, collision samples, and route IDs. Corrective samples enter only
the training split; the existing validation and test routes remain frozen.
