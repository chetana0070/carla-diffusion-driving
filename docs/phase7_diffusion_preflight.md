# Phase 7 temporal diffusion policy preflight

## Objective

Phase 7 introduces a conditional diffusion policy without changing the frozen temporal-BC
or Phase 6 safety baselines. The model predicts 16 future steering/longitudinal actions from
four synchronized image/state observations and the route/traffic-light condition.

## Architecture

- Shared EfficientNet-B0 visual encoder over four camera frames.
- Per-frame state projection and GRU temporal encoder.
- Route-command and traffic-light conditioning.
- Transformer action denoiser with learned action-position embeddings.
- 100-step cosine training schedule.
- Deterministic 10-step DDIM deployment sampler (`eta=0`).
- Receding-horizon contract: predict 16 actions and execute at most four before replanning.

The initial preflight uses batch size four to fit the RTX 5060 Laptop GPU's 8 GiB VRAM.

## Acceptance order

1. Static configuration, unit tests, model/DDIM smoke, and linting pass.
2. One-epoch GPU/data smoke produces a finite checkpoint and report.
3. Full training selects the best validation weighted noise MSE.
4. Offline action metrics are compared with temporal BC, with no claim based on noise MSE
   alone.
5. Batch-one end-to-end DDIM p95 latency is at most 100 ms.
6. Only then is a receding-horizon CARLA runtime added and evaluated behind the frozen Phase
   6 deterministic safety envelope.

## Commands

```bash
python scripts/smoke_phase7_diffusion_model.py

python scripts/train_diffusion_policy.py \
  --smoke \
  --output-dir artifacts/checkpoints/phase7_diffusion_smoke

python scripts/benchmark_diffusion_latency.py \
  --checkpoint artifacts/checkpoints/phase7_diffusion_smoke/best.pt
```

The smoke checkpoint is diagnostic and must not be promoted as a research result.
