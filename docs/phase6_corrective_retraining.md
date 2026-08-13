# Phase 6 corrective retraining

Phase 6.2 assigns seven accepted expert-recovery episodes to training and holds out two
complete correction episodes for recovery validation. The original nominal validation and
test route assignments, rows, and normalization statistics remain frozen.

The merge rejects correction datasets containing non-intervention or collision-labelled
samples. Hidden `.failed` rollout directories are retained for diagnosis but ignored.
Corrective training windows receive a 1.5x multiplier. Recovery-validation windows are not
reweighted. The split is deterministic and episode-level, preventing temporal leakage.

Prepare the merged index:

```bash
python scripts/prepare_phase6_training_data.py
```

Run a one-epoch checkpoint-resume preflight:

```bash
python scripts/train_temporal_bc.py \
  --processed-root data/processed/phase6_corrective_v2 \
  --initial-checkpoint artifacts/checkpoints/phase5_temporal_bc_v080/best.pt \
  --epochs 8 \
  --learning-rate 0.00003 \
  --freeze-encoder-epochs 8 \
  --corrective-validation-split correction_validation \
  --max-nominal-validation-degradation-fraction 0.05 \
  --output-dir artifacts/checkpoints/phase6_dual_gate_v120
```

After the preflight passes, run the full fine-tune in a new output directory with 8 epochs.
Checkpoint selection continues to use the untouched validation split. Closed-loop evaluation,
not offline error alone, remains the promotion gate.

Checkpoint selection minimizes recovery-validation RMSE while limiting nominal-validation
RMSE degradation to 5%. Epoch 0 remains available as the fallback. Closed-loop evaluation,
not offline error alone, remains the promotion gate.
