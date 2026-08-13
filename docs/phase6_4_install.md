# Phase 6.4 hotfix installation and controlled test

Version 1.4.3 corrects acceptance semantics for a preemptive-only run: zero learned
recovery activations now pass the disengagement check when the gate is inactive at episode
end. Any activated recovery without a matching exit, or a gate active at termination, still
fails. This validator-only correction can be applied to the frozen v1.4.2 artifacts without
rerunning CARLA.

## Install

From the project directory:

```bash
unzip -o ~/Downloads/carla-diffusion-driving-phase6.4.2-anticipatory-v1.4.2.zip \
    -d ~/carla-diffusion-driving

cd ~/carla-diffusion-driving
conda activate carla310
chmod +x scripts/validate_phase6_safety_smoke.py
python -m pip install --editable ".[dev]"
```

## Validate source

```bash
python -m ruff check src scripts tests
python -m mypy src
python -m unittest discover -s tests -v
python -m py_compile scripts/*.py src/carla_diffusion/*.py tests/*.py
bash -n scripts/run_phase6_residual_closed_loop.sh
git diff --check
```

## Run the controlled same-seed smoke

The command opens live rendering, records the ego-camera video, stops the owned CARLA
process group after execution, and runs the Phase 6.4 acceptance checker:

```bash
make phase6-safety-smoke
```

If the acceptance checker exits nonzero, do not run the three-seed evaluation. Preserve the
report, telemetry, and video for diagnosis.

## Review the output

```bash
python scripts/summarize_residual_diagnostics.py \
    artifacts/evaluations/phase6_safety_smoke_v142_telemetry/episode-000-seed-20260901.jsonl

xdg-open \
    artifacts/evaluations/phase6_safety_smoke_v142_videos/episode-000-seed-20260901-ego.mp4
```

Close the video player after review:

```bash
pkill -TERM -x vlc || true
```
