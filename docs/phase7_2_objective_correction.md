# Phase 7.2 diffusion objective correction

Phase 7.1 showed that the epoch-7 diffusion checkpoint matched aggregate longitudinal behavior
but failed state-specific accuracy: joint RMSE was 3.12 times temporal BC, and generated action
chunks were substantially less smooth than expert targets. Noise-seed search is therefore closed
as a corrective path.

## Controlled intervention

Phase 7.2 preserves the architecture, dataset splits, DDIM sampler, temporal BC baseline, and
offline promotion thresholds. It changes only the learning and checkpoint-selection contract:

- Warm-start from the disclosed Phase 7 checkpoint.
- Retain weighted noise-prediction loss.
- Reconstruct the clean action chunk at each sampled diffusion timestep.
- Apply Smooth L1 clean-action supervision with 2x longitudinal weighting.
- Apply temporal-derivative supervision to reduce adjacent-action oscillation.
- Select checkpoints by fixed-seed sampled first-action joint RMSE on validation data rather than
  noise MSE.

The objective is:

```text
L = 1.0 L_noise + 1.0 L_clean_action + 0.5 L_temporal_derivative
```

## Gates

1. One-epoch warm-start smoke produces finite component losses and a checkpoint.
2. Full corrective training uses a reduced learning rate and early stopping.
3. The Phase 7.1 evaluator is rerun unchanged against temporal BC.
4. CARLA remains blocked unless every offline promotion gate passes.

## Preflight command

```bash
python scripts/train_diffusion_policy.py \
  --smoke \
  --initial-checkpoint artifacts/checkpoints/phase7_diffusion_v161/best.pt \
  --output-dir artifacts/checkpoints/phase7_objective_smoke_v180
```
