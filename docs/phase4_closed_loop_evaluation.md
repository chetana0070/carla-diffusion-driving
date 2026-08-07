# Phase 4: Closed-Loop Behavioral-Cloning Evaluation

## Decision objective

Determine whether the offline single-frame BC checkpoint can control a CARLA
vehicle without an expert or Traffic Manager controlling the ego actor. This is
a baseline characterization, not a claim that the policy is safe.

## Isolation contract

- The ego vehicle receives control only from the epoch-5 BC checkpoint.
- Traffic Manager controls background vehicles only.
- Evaluation uses seed `20260901`, outside the Phase 2 collection seeds.
- The route command is derived from deterministic route geometry.
- The camera, scalar state, normalization, masks, categorical conditions, and
  action mapping match training.
- A positive longitudinal action maps only to throttle; a negative action maps
  only to brake.
- Policy-pipeline latency includes image conversion, resizing, normalization,
  scalar preparation, host-to-device transfer, inference, and synchronization.
- Twenty startup warmup passes initialize image transforms and CUDA kernels
  before the first timed control action.

## Metrics

- route progress and distance traveled;
- collision, lane-invasion, and red-light events;
- mean/max speed and stationary fraction;
- mean/max lane offset;
- steering and longitudinal action-rate smoothness;
- route-command coverage;
- P50/P95/P99/max end-to-end policy latency.

## Live smoke test

Run one 30-second simulated episode before the three-episode baseline:

```bash
CARLA_RENDER_MODE=live \
PHASE4_EPISODES=1 \
PHASE4_TICKS_PER_EPISODE=300 \
PHASE4_BACKGROUND_VEHICLES=2 \
PHASE4_REPORT=artifacts/evaluations/phase4_closed_loop_smoke.json \
./scripts/run_phase4_closed_loop.sh
```

The smoke test passes infrastructure when the report is written, the checkpoint
epoch is 5, all latency values are finite, and maximum steady-state pipeline
latency remains below 100 ms. Collision or poor route progress is a policy
result, not an infrastructure failure. Collision telemetry records the other
actor type and normal impulse for failure classification.

## Full pilot baseline

```bash
CARLA_RENDER_MODE=offscreen ./scripts/run_phase4_closed_loop.sh
```

The default run evaluates three 600-tick episodes with eight background
vehicles. Results become the closed-loop reference for temporal BC. Do not tune
the single-frame model against these evaluation episodes.
