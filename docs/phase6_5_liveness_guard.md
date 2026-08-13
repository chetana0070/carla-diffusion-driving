# Phase 6.5 bounded liveness guard

The Phase 6.4 three-seed suite completed two routes safely but seed `20260902` stopped after
100 ticks with 0% route progress. Telemetry showed a clear traffic light, no active recovery
or governor, and mean longitudinal output of only 0.037. The learned policy therefore did
not provide enough launch authority to overcome vehicle resistance.

Phase 6.5 adds a stateful liveness guard after the recovery safety envelope. It does not
change the frozen temporal BC or residual checkpoints.

The guard activates only when all conditions hold:

- speed stays at or below 0.10 m/s for ten ticks;
- the traffic light is not red;
- no lead vehicle is within 8 m;
- absolute lane offset is at most 0.35 m;
- absolute heading error is at most 8 degrees;
- the recovery safety envelope is inactive.

While active, longitudinal authority is at least 0.30. The guard exits when speed reaches
1.50 m/s, a safety blocker appears, or 30 active ticks elapse. A 20-tick cooldown prevents
rapid repeated launches. Safety recovery always has priority over liveness.

Run the frozen failing-seed smoke:

```bash
make phase6-liveness-smoke
```

The runner uses `PHASE6_RESIDUAL_SEED` and forwards it explicitly as the evaluator's
`--seed` argument. The report must show seed `20260902` and spawn index `132`; any other
seed means the targeted test is invalid even if its safety checker passes.

The command opens live rendering, records video, stops its owned CARLA process, and validates
the same physical, latency, recovery-disengagement, and liveness-disengagement gates used in
Phase 6.4. Do not repeat the full three-seed evaluation unless this targeted smoke passes.

## Phase 6.5.2 global speed envelope

The v1.5.1 correct-seed rollout proved that liveness was restored, but the policy reached
5.42 m/s while lane offset remained just below the preemptive threshold. Speed limiting was
therefore coupled too tightly to lane-recovery activation. Version 1.5.2 adds an always-on
deployment governor after recovery and liveness arbitration:

- below 4.0 m/s: preserve the selected longitudinal action;
- 4.0--4.25 m/s: remove positive longitudinal authority;
- 4.25--4.75 m/s: apply at least -0.35 braking;
- at or above 4.75 m/s: apply at least -0.70 emergency braking.

The validator now computes the 5.0 m/s gate over every telemetry row. This change does not
modify either learned checkpoint and does not interfere with the stationary launch guard.

## Phase 6.5.3 arbitration correction

The v1.5.2 rollout stayed below 5.0 m/s but stopped after 148 ticks. Telemetry showed a
persistent 0.24 m lane offset: enough to activate preemptive deterministic steering, but
not enough to activate learned recovery. Treating all preemptive safety as a liveness block
left the stationary counter at zero indefinitely.

The corrected priority order is:

1. active learned recovery and its cooldown block liveness;
2. preemptive-only steering correction may coexist with liveness when the guard's own lane,
   heading, signal, and lead-distance checks pass;
3. the global speed envelope applies last and retains final longitudinal authority.

This is an arbitration change only; all learned checkpoints and frozen acceptance limits
remain unchanged.
