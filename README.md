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
validation. No model is trained in these infrastructure phases.

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
