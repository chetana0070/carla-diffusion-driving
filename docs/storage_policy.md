# Storage Policy

The local Ubuntu partition is treated as constrained research infrastructure.

## Controls

- Collection may start only with at least 100 GiB free.
- Collection stops automatically at 80 GiB free.
- RGB uses JPEG quality 85 at 640 x 360.
- Dataset shards are capped at 2 GiB.
- Frames are never duplicated for temporal windows.
- Raw, processed, checkpoint, and video artifacts are ignored by Git.
- Only the active dataset generation and latest validated checkpoint remain local.
- Completed shards move to ASU research storage before the next collection batch.

## Expected pilot budget

| Item | Budget |
|---|---:|
| CARLA packaged installation | 20–30 GiB |
| Python environments | 10–20 GiB |
| Pilot demonstrations | 10–20 GiB |
| Checkpoints, logs, videos | 10 GiB |
| Reserved safety headroom | 80 GiB |

The Phase 0 configuration deliberately prevents starting a collection job below
the 100 GiB threshold.

