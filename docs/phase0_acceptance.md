# Phase 0 Acceptance Gate

## Required before CARLA installation

- [ ] `python -m unittest discover -s tests -v` passes.
- [ ] `python scripts/validate_config.py` passes.
- [ ] Python 3.10 environment is active.
- [ ] NVIDIA GPU is visible with at least 8 GiB VRAM.
- [ ] Ports 2000 and 2001 are available.
- [ ] At least 100 GiB disk space is free.
- [ ] Observation/action schema is unchanged or version-bumped.
- [ ] Town splits are disjoint.
- [ ] Git ignores raw data, checkpoints, videos, and secrets.

## Compatibility risk requiring Phase 1 evidence

The development host uses Ubuntu 24.04 and kernel 7.x, while the official CARLA
0.9.16 packaged targets are Ubuntu 20.04 and 22.04. This is a controlled risk.
The Phase 1 gate requires:

1. packaged server launch at Low quality;
2. successful Python client connection;
3. synchronous 10 Hz ticking for 1,000 ticks;
4. 1,000 RGB frames with no missing or duplicate frame IDs;
5. clean shutdown and deterministic replay under the same seed.

Failure routes the simulator into an Ubuntu 22.04 GPU-enabled container; it does
not trigger a source build.

