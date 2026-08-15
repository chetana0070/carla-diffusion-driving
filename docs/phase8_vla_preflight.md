# Phase 8.0 — Hierarchical VLA Preflight

## Decision

Phase 8 begins with a compact, language-conditioned action planner rather than a
large end-to-end VLM directly controlling CARLA. The planner runs at 2 Hz and emits
a bounded 16-action chunk. The runtime executes five 10 Hz actions and then requires
a new plan. Phase 6 safety, speed, and liveness arbitration remains the final control
authority on every simulator tick.

This design isolates three different responsibilities:

1. The VLA planner interprets visual context, vehicle state, and an instruction.
2. The hierarchical executor enforces replanning cadence and action bounds.
3. Deterministic arbitration protects the existing closed-loop safety envelope.

## Claim boundary

The first dataset uses deterministic language generated from existing route-command
and traffic-light metadata. It supports supervised conditioning, reproducibility, and
runtime integration. It does **not** demonstrate free-form instruction following,
semantic scene reasoning, causal explanation, or open-vocabulary grounding.

Those claims require independently authored language, scenario diversity, counterfactual
instruction tests, and held-out semantic evaluation in a later Phase 8 milestone.

## Frozen contracts

- Observation history: 4 frames at 10 Hz.
- Image input: 224 × 224 RGB.
- State input: 10 normalized values, including availability masks.
- Instruction width: 20 tokens using a versioned project vocabulary.
- Planner horizon: 16 steering/longitudinal actions.
- Planner rate: 2 Hz.
- Executed prefix: 5 actions before mandatory replanning.
- Action range: both axes clipped to `[-1, 1]`.
- Dataset split: inherited route-level train/validation/test assignment.
- Primary future system target: safety-active fraction below 60%, compared with the
  Phase 7.5 baseline of 91.78%, without weakening any safety-event budget.

## Local preflight

The Phase 3 processed dataset and its source images must remain available locally.

```bash
conda activate carla310
cd ~/carla-diffusion-driving

make phase8-vla-prepare
make phase8-vla-validate
make phase8-vla-model-smoke
```

The preparation report must disclose its structured language source and produce a
SHA-256 digest for `vla_windows.jsonl`. The validator rejects route leakage, malformed
instruction encodings, inconsistent horizons, unbounded actions, and digest mismatch.

## Training preflight

Use a new output directory for every experiment:

```bash
python scripts/train_phase8_vla.py \
    --smoke \
    --output-dir artifacts/checkpoints/phase8_vla_smoke_v220
```

Full local training is available but SOL is preferred:

```bash
make phase8-vla-train
```

The checkpoint report remains an offline imitation-learning artifact. It cannot be
promoted to CARLA until latency, untouched-test performance, instruction sensitivity,
and closed-loop intervention-dependency gates are implemented and passed.

## SOL workflow

Copy `data/processed/phase8_vla_v1`, the underlying raw image dataset, and the updated
repository to scratch storage. ASU Research Computing recommends the system-provided
Mamba environment rather than installing a separate Conda distribution. The included
SBATCH file requests one public A100 in the `htc` partition:

```bash
cd /scratch/cchakrap/carla-diffusion-driving-v2.0.0/repository
export PHASE8_PROJECT_ROOT="$PWD"
export PHASE8_ENVIRONMENT_NAME=carla310
export PHASE8_DATASET_ROOT="$PWD/data/raw/phase2_pilot_v2_v043"
sbatch scripts/slurm/train_phase8_vla.sbatch
squeue -u "$USER"
```

Confirm live partition and GPU availability with `sinfo` before submission. Scheduler
guidance: <https://docs.rc.asu.edu/requesting-resources/>.

## Promotion sequence

1. Validate deterministic language-conditioned data.
2. Pass a synthetic forward/backward model smoke test.
3. Train on SOL and freeze the best validation checkpoint.
4. Measure untouched-test action-chunk accuracy and instruction sensitivity.
5. Benchmark planner latency locally on the deployment GPU.
6. Run one live-rendered, video-enabled CARLA smoke test.
7. Run the frozen three-seed suite and compare arbitration dependency against Phase 7.5.

No safety threshold may be relaxed to promote the VLA.
