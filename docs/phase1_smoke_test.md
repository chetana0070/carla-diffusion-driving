# Phase 1: Packaged Simulator and Sensor Smoke Test

## Version decision

CARLA 0.9.16/UE4.26 is pinned for the experiment. CARLA 0.10.0 migrates to
UE5.5 and publishes a materially higher recommended hardware profile. Version
0.9.16 is the appropriate controlled baseline for the RTX 5060 Laptop GPU with
8 GB VRAM and preserves Scenario Runner 0.9.16 compatibility.

## Installation

```bash
conda activate carla310
cd /home/chetana/carla-diffusion-driving
chmod +x scripts/install_carla_0916.sh scripts/run_phase1_smoke.sh
./scripts/install_carla_0916.sh
```

The installer resolves the official GitHub release asset, resumes interrupted
downloads, verifies the release digest when GitHub exposes one, extracts the
packaged simulator to `/home/chetana/opt/carla-0.9.16`, deletes the downloaded
archive, and installs the matching Python API.

## Smoke test

```bash
conda activate carla310
cd /home/chetana/carla-diffusion-driving
./scripts/run_phase1_smoke.sh
```

The runner starts the server at Low quality with off-screen rendering, waits for
port 2000, performs 1,000 synchronous ticks at 10 Hz simulation time, and stops
the server automatically.

## Passing contract

- exactly 1,000 camera frames received;
- every camera frame ID equals its world tick ID;
- frame IDs are unique and consecutive;
- every image is 640 x 360;
- client and server versions match;
- actors are destroyed and asynchronous settings restored;
- JSON report status is `passed`.

Outputs:

- `artifacts/evaluations/phase1_smoke_report.json`;
- `artifacts/evaluations/carla_server_phase1.log`.

If the server fails to launch on Ubuntu 24.04/kernel 7.x, preserve the server
log and route to the Ubuntu 22.04 container contingency. Do not compile CARLA
from source.

