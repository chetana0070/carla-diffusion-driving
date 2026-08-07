# Phase 5.2: Closed-Loop Failure Diagnosis

## Decision objective

Determine whether Phase 5 failed because of the learned policy or because the
route-generation and evaluation harness cannot support successful driving. No
new model is trained in this phase.

## Controlled comparison

The learned temporal policy and CARLA Traffic Manager expert use the same:

- Town01 map;
- seeds 20260901 through 20260903;
- spawn points and generated waypoint routes;
- 600-tick limit at 10 Hz;
- eight background vehicles;
- collision and no-progress termination rules.

The expert oracle passes only when mean route progress is at least 25% and it
records zero collisions. This threshold validates meaningful route traversal;
it is not the final autonomous-driving success target.

## Telemetry contract

Each episode writes one JSON object per tick containing:

- frame, seed and route index;
- route progress and cumulative distance;
- speed and lane offset;
- route command and traffic-light state;
- steering, longitudinal action, throttle and brake;
- collision contact and no-progress state;
- policy pipeline latency.

An episode terminates as `no_progress` after traveling less than two meters
across a 100-tick window. The monitor resets during observed red lights so a
legal stop is not classified as policy failure. This prevents a stationary
policy from appearing safe merely because it avoided collisions.

## Execution sequence

First rerun the temporal policy under the diagnostic contract:

```bash
CARLA_RENDER_MODE=offscreen \
PHASE5_EPISODES=3 \
PHASE5_TICKS_PER_EPISODE=600 \
PHASE5_BACKGROUND_VEHICLES=8 \
PHASE5_REPORT=artifacts/evaluations/phase5_policy_diagnostic_v090.json \
PHASE5_TELEMETRY_DIR=artifacts/evaluations/phase5_policy_telemetry_v090 \
./scripts/run_phase5_closed_loop.sh
```

Then run one live expert smoke test:

```bash
CARLA_RENDER_MODE=live \
PHASE5_EPISODES=1 \
PHASE5_TICKS_PER_EPISODE=300 \
PHASE5_BACKGROUND_VEHICLES=2 \
PHASE5_EXPERT_REPORT=artifacts/evaluations/phase5_expert_smoke_v090.json \
PHASE5_EXPERT_TELEMETRY_DIR=artifacts/evaluations/phase5_expert_smoke_telemetry_v090 \
./scripts/run_phase5_expert_oracle.sh
```

The runner owns CARLA in a dedicated process group and closes the server and
live window automatically after success, failure, or interruption. It refuses
to start when port 2000 is already occupied, preventing an apparently live run
from silently connecting to a stale off-screen server.

Video is disabled by default. When visual evidence is required, add:

```bash
PHASE5_SAVE_VIDEO=1 \
PHASE5_VIDEO_DIR=artifacts/evaluations/phase5_expert_videos \
./scripts/run_phase5_expert_oracle.sh
```

This writes one disk-efficient 10 Hz MP4 per episode from the policy's ego RGB
camera. The MP4 is independent of the live window, so evidence can also be
recorded during faster off-screen evaluation. `ffmpeg` is required only when
video capture is enabled.

If it passes technically, run the frozen expert evaluation:

```bash
CARLA_RENDER_MODE=offscreen \
PHASE5_EPISODES=3 \
PHASE5_TICKS_PER_EPISODE=600 \
PHASE5_BACKGROUND_VEHICLES=8 \
PHASE5_EXPERT_REPORT=artifacts/evaluations/phase5_expert_oracle_v090.json \
PHASE5_EXPERT_TELEMETRY_DIR=artifacts/evaluations/phase5_expert_telemetry_v090 \
./scripts/run_phase5_expert_oracle.sh
```

Generate the decision report:

```bash
python scripts/summarize_phase5_diagnostics.py
```

## Decision gate

- `policy_failure_confirmed_collect_corrections`: the expert passes; proceed to
  recovery-state and DAgger-style corrective collection.
- `route_or_oracle_setup_requires_correction`: the expert fails; repair the
  route harness before collecting or training more data.

Diffusion training remains blocked until this gate is resolved.
