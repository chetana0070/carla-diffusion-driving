# Frozen Project Specification

## Objective

Build and evaluate a deployable temporal driving policy in CARLA through the
sequence: single-frame BC -> temporal BC -> diffusion policy -> DAgger ->
bounded residual SAC.

## Design constraints

1. Closed-loop route performance is the primary outcome.
2. Complete routes—not individual frames—form dataset splits.
3. Deployment observations may not include privileged failure labels.
4. Steering and longitudinal acceleration form the only learned action.
5. Throttle/brake arbitration is deterministic downstream control logic.
6. Every experiment is config-addressed and seed-controlled.
7. RL cannot begin until the temporal supervised policy completes basic routes.

## State vector order

The eight continuous state values are frozen in this order:

1. ego speed in m/s;
2. longitudinal acceleration in m/s^2;
3. speed limit in m/s;
4. signed lane-center offset in meters;
5. signed heading error in radians;
6. lead-vehicle distance in meters (`-1` when unavailable);
7. lead-vehicle relative speed in m/s (`0` when unavailable);
8. traffic-light stop-line distance in meters (`-1` when unavailable).

Traffic-light state and high-level route command remain categorical fields.

## Output contract

`steering` and `longitudinal_acceleration` are normalized to `[-1, 1]`.
Positive longitudinal acceleration maps to throttle; negative values map to
brake. Throttle and brake cannot be active simultaneously.

## Dataset unit

One episode contains metadata, synchronized sensor frames, state/action rows,
and event annotations. Frames are stored once. Four-frame histories and
sixteen-action chunks are assembled by the dataset loader.

## Frozen comparison set

1. CARLA expert agent.
2. Single-frame BC.
3. Temporal BC.
4. Temporal diffusion policy.
5. Diffusion + DAgger.
6. Diffusion + DAgger + residual SAC.

