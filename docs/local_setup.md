# Local Phase 0 Setup

## Host snapshot

- Intel Core i7-13620H, 10 cores / 16 threads;
- NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB VRAM, 60 W;
- 16 GB physical RAM;
- NVIDIA driver 580.126.20 with CUDA driver capability 13.0;
- Ubuntu 24.04 with kernel 7.0.0-28-generic;
- approximately 128 GiB free after archival and environment cleanup.

## Bootstrap

From the repository root:

```bash
chmod +x scripts/bootstrap_phase0.sh
./scripts/bootstrap_phase0.sh
```

The script creates a dedicated `carla310` environment, installs only the
dependency-light development package, runs the tests, and writes a read-only
system report. It does not download or install CARLA.

Activate the environment in future terminals with:

```bash
conda activate carla310
```

## Expected result on this host

- Python compatibility: pass under `carla310`;
- disk floor: pass;
- GPU and VRAM: pass;
- ports: pass when no CARLA server is running;
- RAM minimum: pass narrowly;
- Ubuntu official-target check: warning;
- recommended RAM and full-project disk checks: warning.

Warnings are design constraints. Blocking failures must be fixed before Phase 1.

