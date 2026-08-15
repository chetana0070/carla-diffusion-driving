# Phase 7.5 — Factorized Policy Closed-Loop Evaluation

## Decision boundary

Phase 7.4 passed the frozen offline and latency gates. Phase 7.5 is the first
CARLA deployment of that checkpoint. Offline promotion does not imply driving
success; this phase freezes a separate three-seed closed-loop decision.

The deployed controller is explicitly factorized:

- diffusion generates the 16-step steering plan;
- the deterministic temporal head generates the longitudinal plan;
- the first four actions are executed before replanning;
- a route-command or traffic-light transition forces immediate replanning;
- the frozen Phase 6 deterministic safety, liveness, and speed arbitration is
  applied on every control tick.

Telemetry preserves the base factorized actions and every arbitration event.
Results must therefore be reported as **factorized policy with deterministic
safety arbitration**, not as an unassisted diffusion policy.

## Installation

Install the patch from `~/Downloads` and retain the released checkpoint:

```bash
cd ~/Downloads
unzip -o carla-diffusion-driving-phase7.5-v2.1.0.zip \
  -d ~/carla-diffusion-driving

cd ~/carla-diffusion-driving
conda activate carla310
chmod +x \
  scripts/evaluate_factorized_closed_loop.py \
  scripts/evaluate_factorized_seeded_closed_loop.py \
  scripts/run_phase7_factorized_closed_loop.sh \
  scripts/validate_phase7_factorized_closed_loop.py

python -m pip install --editable ".[dev]"
python -m ruff check src scripts tests
python -m mypy src
python -m unittest discover -s tests -v
git diff --check
```

The evaluator requires:

```text
artifacts/checkpoints/phase7_factorized_longitudinal_v200/best.pt
data/processed/phase3_pilot_v1/normalization.json
```

## Controlled smoke

The smoke uses seed `20260901`, 100 ticks, two background vehicles, live
rendering, telemetry, and one MP4. It validates integration only and is not a
promotion result.

```bash
make phase7-factorized-closed-loop-smoke
```

The CARLA window is owned by the launcher and closes automatically on success,
failure, Ctrl-C, or termination.

## Frozen three-seed suite

```bash
CARLA_RENDER_MODE=live \
PHASE7_SAVE_VIDEO=1 \
make phase7-factorized-closed-loop
```

This runs seeds `20260901`, `20260902`, and `20260903` for 300 ticks each with
eight background vehicles. To run without a visible window or videos:

```bash
CARLA_RENDER_MODE=offscreen PHASE7_SAVE_VIDEO=0 \
make phase7-factorized-closed-loop
```

Validate only after the complete three-seed suite:

```bash
make phase7-factorized-validate
```

## Frozen acceptance contract

- exact released checkpoint SHA-256;
- exactly three expected seeds and complete telemetry;
- 300 ticks per seed;
- zero collisions, lane invasions, red-light violations, and no-progress
  terminations;
- mean route progress at least 9%;
- mean distance at least 80 m;
- maximum per-episode stationary fraction of 60%;
- maximum speed of 5.0 m/s and lane offset of 1.5 m;
- maximum first-tick latency of 200 ms;
- maximum steady-state tick latency of 100 ms;
- no recovery or liveness guard left active at episode termination.

The validator writes
`artifacts/evaluations/phase7_factorized_closed_loop_v210_acceptance.json` and
returns nonzero when any gate fails. Do not relax thresholds after observing
the three-seed result; freeze failures and diagnose them as a new phase.
