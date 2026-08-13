# Phase 6.4.2 anticipatory safety-gated residual recovery

Direct DAgger fine-tuning reduced recovery-validation RMSE but exceeded the nominal-driving
regression budget. Phase 6.3 therefore freezes temporal BC and trains a small bounded residual
only on accepted expert-recovery windows.

At runtime, the base action remains unchanged until lane offset or heading error crosses the
same observable geometric thresholds used for collection. Phase 6.3 diagnostics showed that
the learned residual initially reduced lane offset but remained active after crossing lane
center, added throttle, and produced an unsafe high-speed overshoot.

Phase 6.4 preserves the frozen checkpoint and adds deterministic deployment constraints.
The v1.4.1 controlled run completed all 300 ticks without a collision, but entered recovery
at 7.01 m/s, peaked at 7.12 m/s, and recorded two lane invasions. Its eight recovery cycles
also showed repeated center crossings caused by a fixed steering override.

Version 1.4.2 therefore changes only the deterministic arbitration layer:

- recovery can exit after three active ticks;
- center crossing inside 0.35 m ends recovery immediately;
- three stable centered ticks also end recovery;
- recovery is limited to 20 ticks, followed by a five-tick cooldown;
- residual steering is rejected when it points away from signed lane center;
- the safety envelope begins preemptively at 0.15 m lane offset or 4 degrees heading error,
  before the learned residual's original 0.45 m / 10 degree activation thresholds;
- final steering is guaranteed to point toward lane center outside the direction-guard
  boundary, with proportional authority from 0.03 to 0.15 rather than a fixed 0.15 command;
- positive longitudinal residual is suppressed;
- recovery speed is governed at 4 m/s, braking begins at 5 m/s, and emergency braking
  begins at 7 m/s;
- absolute steering is capped at 0.45 and steering changes are limited to 0.20 per tick.

The speed governor and steering slew limit remain active during preemption and cooldown so
the policy cannot accelerate into a developing recovery or immediately undo one. The learned
checkpoint, route, seed, background traffic, and acceptance thresholds remain frozen.

Train:

```bash
python scripts/train_residual_correction.py
```

Run one live smoke episode, optionally saving video:

```bash
CARLA_RENDER_MODE=live \
PHASE6_RESIDUAL_EPISODES=1 \
PHASE6_RESIDUAL_TICKS_PER_EPISODE=300 \
PHASE6_RESIDUAL_BACKGROUND_VEHICLES=2 \
PHASE6_RESIDUAL_SAVE_VIDEO=1 \
./scripts/run_phase6_residual_closed_loop.sh
```

The owned CARLA process group is closed automatically. Promotion requires lower intervention
or collision frequency than Phase 5 across the frozen three-seed route suite.

The first test must reuse seed `20260901`, 300 ticks, and two background vehicles. Promotion
to the three-seed suite requires 300 ticks without collision, recovery speed no greater than
5 m/s, maximum lane offset below 1.5 m, no more than one lane invasion, and latency below
100 ms after tick 0. Tick 0 is reported separately under a disclosed 200 ms cold-start gate.
