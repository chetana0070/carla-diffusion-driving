# CARLA Diffusion Driving

Research-grade rebuild of an end-to-end CARLA driving policy using behavioral
cloning, temporal action diffusion, DAgger corrections, and bounded residual
SAC refinement.

## Current milestone

Phase 0 freezes the experimental contract before simulator installation:

- versioned observation/action schema;
- deterministic simulator and dataset configuration;
- route-level train/validation/test split policy;
- system and configuration validators;
- experiment matrix and acceptance gates;
- storage controls for a 240 GB Ubuntu partition.

Phase 0 passed on the target laptop. Phase 1 installed the pinned packaged
server and passed a deterministic 1,000-tick RGB synchronization test. Phase 2
adds atomic expert-episode collection, storage gates, and full dataset
validation. Phase 3 converts the accepted pilot into leakage-resistant temporal
windows and training-only normalization statistics. No model is trained in
these infrastructure phases.

## Research question

Can a temporal diffusion driving policy, initialized from expert
demonstrations and improved with DAgger plus bounded residual SAC, outperform a
temporal behavioral-cloning baseline on unseen CARLA routes without degrading
safety or control latency?

## Frozen policy interface

- Observation history: 4 front RGB frames.
- Capture resolution: 640 x 360; model input: 224 x 224.
- Control frequency: 10 Hz.
- Additional state: speed, longitudinal acceleration, speed limit, lane offset,
  heading error, lead distance, lead relative speed, traffic-light distance,
  traffic-light state, and route command.
- Action: `[steering, longitudinal_acceleration]`, each in `[-1, 1]`.
- Diffusion horizon: 16 actions; execute 4 actions, observe, and replan.

Collision, lane invasion, red-light violation, and route completion are labels,
reward signals, and metrics—not deploy-time policy inputs.

## Quick verification

The Phase 0 suite uses only the Python standard library:

```bash
python -m unittest discover -s tests -v
python scripts/validate_config.py
python scripts/validate_system.py --json-out artifacts/system_report.json
```

The system validator is expected to warn on Ubuntu 24.04/kernel 7.x because the
official CARLA 0.9.16 packaged target is Ubuntu 20.04/22.04. The project may
proceed only after the Phase 1 simulator smoke test passes.

## Repository layout

```text
configs/                 Frozen simulator, dataset, model, and evaluation config
schemas/                 Versioned observation/action/data contract
src/carla_diffusion/     Dependency-light core validation code
scripts/                 User-facing validation commands
tests/                   Phase 0 acceptance tests
docs/                    Specification, storage policy, experiment plan
data/                    Local-only datasets (ignored by Git)
artifacts/               Local-only checkpoints and evaluation output
```

## Phase 1 entry gate

Proceed to CARLA installation only when:

1. all unit tests pass;
2. configuration validation reports no errors;
3. at least 100 GiB remains free;
4. Python 3.10 is active;
5. ports 2000 and 2001 are available;
6. the NVIDIA GPU is visible.

See `docs/phase0_acceptance.md` for the complete gate.

## Phase 1 commands

```bash
conda activate carla310
./scripts/install_carla_0916.sh
./scripts/run_phase1_smoke.sh
```

CARLA 0.9.16/UE4.26 is pinned intentionally. CARLA 0.10.0/UE5.5 has a larger
recommended hardware envelope than the target 8 GB laptop GPU.

## Phase 2 pilot

```bash
conda activate carla310
./scripts/run_phase2_pilot.sh
```

The pilot records ten seeded Town01 Traffic Manager episodes (6,000 synchronized
samples), validates the frozen schema and every image reference, and refuses to
start below 100 GiB free. See `docs/phase2_data_pilot.md` for the exit gate.

For visual debugging, prefix either runner with `CARLA_RENDER_MODE=live`. This
opens a Low-quality chase-camera window; automated and benchmark execution stays
off-screen by default.

Audit the completed pilot before expanding collection:

```bash
python scripts/audit_phase2_dataset.py data/raw/phase2_pilot
```

The audit measures temporal-window yield, control and state coverage, route
commands, traffic context, safety events, JPEG integrity, and per-episode
balance. It produces JSON, Markdown, CSV, and a labeled contact sheet under
`artifacts/evaluations/`.

## Phase 3 data preparation

```bash
python scripts/prepare_phase3_data.py \
    data/raw/phase2_pilot_v2_v043 \
    --output-root data/processed/phase3_pilot_v1
```

The preparation stage creates 4-frame/16-action windows, an 8/1/1 pilot
route split, training-only normalization statistics, explicit lead/light
availability masks, acceleration clipping metadata, and sampling weights for
rare maneuvers and redundant stationary brake holds. Raw Phase 2 files are
never modified.

## Phase 4 single-frame BC baseline

PyTorch 2.11.0 and torchvision 0.26.0 are pinned for the target RTX 5060 CUDA
12.8 environment. Verify the architecture and gradients first, then run a
two-batch data/training smoke test:

```bash
python scripts/smoke_phase4_model.py
python scripts/train_single_frame_bc.py \
    --smoke \
    --output-dir artifacts/checkpoints/phase4_single_frame_smoke
```

If both pass, train the ImageNet-initialized single-frame baseline:

```bash
python scripts/train_single_frame_bc.py --pretrained
```

This baseline uses only the current frame and predicts the first expert action.
It establishes the controlled comparison for temporal BC and temporal action
diffusion. See `docs/phase4_single_frame_bc.md` for the metric and checkpoint
selection contract.

Evaluate the selected checkpoint under actual CARLA feedback before beginning
the temporal model:

```bash
CARLA_RENDER_MODE=live \
PHASE4_EPISODES=1 \
PHASE4_TICKS_PER_EPISODE=300 \
PHASE4_BACKGROUND_VEHICLES=2 \
PHASE4_REPORT=artifacts/evaluations/phase4_closed_loop_smoke.json \
./scripts/run_phase4_closed_loop.sh
```

The ego vehicle is controlled only by the learned policy. The evaluator records
route progress, safety events, control smoothness, and end-to-end policy latency.

## Phase 5 temporal BC

The next ablation replaces the single current frame with four image/state
observations while retaining a single current-action target:

```bash
python scripts/smoke_phase5_temporal_model.py
python scripts/train_temporal_bc.py \
    --smoke \
    --output-dir artifacts/checkpoints/phase5_temporal_smoke_v080
```

The shared EfficientNet-B0 and GRU model uses the same Phase 3 windows, weights,
route split, metric definitions, and checkpoint-selection rule as single-frame
BC. This isolates temporal context before action-chunk diffusion is introduced.

Benchmark and evaluate the selected temporal checkpoint using the frozen Phase 4
route contract:

```bash
python scripts/benchmark_temporal_bc_latency.py

CARLA_RENDER_MODE=live \
PHASE5_EPISODES=1 \
PHASE5_TICKS_PER_EPISODE=300 \
PHASE5_BACKGROUND_VEHICLES=2 \
PHASE5_REPORT=artifacts/evaluations/phase5_closed_loop_smoke_v081.json \
./scripts/run_phase5_closed_loop.sh
```

The temporal runtime repeats the first synchronized observation to initialize
its four-frame buffer, then rolls forward one frame per control tick. The buffer
is reset between episodes to prevent route leakage.

## Phase 5.2 failure diagnosis

Before collecting corrective demonstrations, rerun the frozen temporal policy
with telemetry/no-progress detection, then validate the same routes with CARLA
Traffic Manager under the identical seeds and traffic load:

```bash
CARLA_RENDER_MODE=offscreen \
PHASE5_EPISODES=3 \
PHASE5_TICKS_PER_EPISODE=600 \
PHASE5_BACKGROUND_VEHICLES=8 \
PHASE5_REPORT=artifacts/evaluations/phase5_policy_diagnostic_v090.json \
PHASE5_TELEMETRY_DIR=artifacts/evaluations/phase5_policy_telemetry_v090 \
./scripts/run_phase5_closed_loop.sh
```

Then smoke-test the expert oracle:

```bash
CARLA_RENDER_MODE=live \
PHASE5_EPISODES=1 \
PHASE5_TICKS_PER_EPISODE=300 \
PHASE5_BACKGROUND_VEHICLES=2 \
PHASE5_EXPERT_REPORT=artifacts/evaluations/phase5_expert_smoke_v090.json \
PHASE5_EXPERT_TELEMETRY_DIR=artifacts/evaluations/phase5_expert_smoke_telemetry_v090 \
./scripts/run_phase5_expert_oracle.sh
```

Every episode writes compact tick-level JSONL telemetry. Runs terminate as
`no_progress` when they travel less than two meters across 100 ticks. After the
three-seed expert run, `scripts/summarize_phase5_diagnostics.py` verifies the
map/seed contract and decides whether the route harness or learned policy is the
primary blocker. The expert capability gate is versioned and combines route
progress, physical distance, safety, and liveness so legal signal stops cannot
dominate a single progress threshold. See `docs/phase5_2_failure_diagnosis.md`.

## Phase 6 corrective demonstrations

The expert-oracle comparison confirms that temporal BC—not the CARLA harness—is
the primary closed-loop blocker. Phase 6 therefore rolls out temporal BC and
transfers the same vehicle to Traffic Manager when observable lane, heading,
lead-vehicle, or liveness thresholds fire. Only contiguous expert recovery
segments are published:

```bash
./scripts/run_phase6_corrections.sh
python scripts/audit_phase6_corrections.py
```

Corrective data remains training-only. Frozen validation and test routes are
not expanded or relabeled. See `docs/phase6_corrective_collection.md`.

## Phase 6 corrective retraining

After triggered expert-recovery collection passes its audit, merge the accepted
corrections into training only and resume from the frozen Phase 5 checkpoint:

```bash
python scripts/prepare_phase6_training_data.py
make phase6-train-preflight
```

The merge preserves the original validation/test rows and normalization exactly,
rejects collision-labelled or non-intervention samples, and excludes hidden failed
rollouts. Two complete correction episodes form a separate recovery-validation
split. Promotion requires recovery improvement while nominal validation degradation
remains within 5%. See `docs/phase6_corrective_retraining.md`.

When direct fine-tuning cannot satisfy both gates, train a bounded recovery residual while
keeping temporal BC frozen:

```bash
make phase6-residual-train
make phase6-residual-closed-loop
```

The residual is activated only by a hysteretic geometric recovery gate. Normal policy output
is unchanged. See `docs/phase6_gated_residual.md`.

Phase 6.4 adds a deterministic recovery safety envelope after the Phase 6.3 residual improved
short-horizon progress but remained active after crossing lane center. The envelope limits
recovery duration, adds cooldown, rejects directionally unsafe steering, suppresses added
throttle, governs recovery speed, and limits steering slew. The residual checkpoint remains
frozen. Run `make phase6-safety-smoke` before any three-seed evaluation.

Phase 6.4.2 extends that arbitration after v1.4.1 completed 300 ticks without collision but
entered recovery above the speed gate and recorded two lane invasions. The deterministic
safety envelope now starts at 0.15 m lane offset or 4 degrees heading error, before the
learned residual activates, and its centering authority scales from 0.03 to 0.15 with lane
error. Recovery remains bounded to 20 ticks. Latency promotion reports the first-observation
cold start separately from steady-state inference.

Phase 6.5 addresses the remaining seed-specific liveness failure. When the vehicle remains
below 0.10 m/s for ten ticks with no red light, no close lead vehicle, safe lane geometry,
and no active safety recovery, a bounded launch guard applies at least 0.30 longitudinal
authority. It releases at 1.50 m/s, after 30 active ticks, or immediately when any safety
blocker appears, then enforces a 20-tick cooldown. Run `make phase6-liveness-smoke` on the
previously failing seed before repeating the three-seed suite.

Phase 6.5.2 decouples the deployment speed envelope from lane-recovery activation. The
policy now coasts from 4.0 m/s, brakes from 4.25 m/s, and applies emergency braking from
4.75 m/s on every tick. This preserves the low-speed liveness launch while adding one-step
margin beneath the frozen 5.0 m/s acceptance ceiling. The validator measures peak speed
across the complete rollout rather than only safety-active rows.

Phase 6.5.3 resolves a liveness arbitration deadlock found on seed `20260902`. Moderate
lane error can keep preemptive steering safety active without activating learned recovery.
The launch guard may now operate concurrently with that deterministic steering correction,
but remains blocked by active learned recovery, recovery cooldown, red lights, close lead
vehicles, or unsafe geometry. The speed envelope remains the final longitudinal authority.

Phase 7 adds the first temporal diffusion-policy implementation. It consumes the same four
observation frames as temporal BC, but denoises a 16-action steering/longitudinal sequence
with a cosine schedule and deterministic 10-step DDIM inference. This milestone intentionally
stops before CARLA deployment: model, dataset, GPU training, sampled-action quality, and p95
latency must pass before the policy is connected to the frozen Phase 6 safety envelope. See
`docs/phase7_diffusion_preflight.md`.
