# Phase 2: Deterministic Expert-Data Pilot

## Objective

Validate the complete expert-data path before scaling collection. The pilot
records ten deterministic Traffic Manager episodes in Town01, each containing
600 synchronized 10 Hz samples (ten simulated minutes and 6,000 total samples).

## Data contract

Every sample contains:

- one 640 x 360 JPEG front-camera image;
- the frozen eight-element continuous state vector;
- traffic-light state and route command;
- Traffic Manager steering and unified throttle-minus-brake action;
- collision, lane-invasion, red-light-violation, and intervention labels.

Episodes are written under `data/raw/phase2_pilot`. A hidden staging directory
is used while collection is active. The directory is renamed into place only
after all rows and images have been flushed successfully. Failed episodes stay
hidden with a failure report and are never treated as training data.

## Safety and storage controls

- Collection cannot start below 100 GiB free.
- Free storage is checked every 100 simulation ticks.
- Collection aborts at or below 80 GiB free.
- Frames are JPEG-encoded once and referenced by relative path.
- Raw data and generated reports remain excluded from Git.

## Run

First prove the runtime integration with one 100-frame episode:

```bash
conda activate carla310
cd /home/chetana/carla-diffusion-driving
PHASE2_EPISODES=1 \
PHASE2_TICKS_PER_EPISODE=100 \
PHASE2_BACKGROUND_VEHICLES=2 \
PHASE2_DATASET_ROOT=data/raw/phase2_smoke \
PHASE2_REPORT_PATH=artifacts/evaluations/phase2_smoke_report.json \
./scripts/run_phase2_pilot.sh
```

After that passes, run the frozen pilot:

```bash
conda activate carla310
cd /home/chetana/carla-diffusion-driving
./scripts/run_phase2_pilot.sh
```

Either the smoke test or full pilot can be observed live by setting the render
mode before the command:

```bash
CARLA_RENDER_MODE=live ./scripts/run_phase2_pilot.sh
```

Live mode opens a 1280 x 720 Low-quality CARLA window and tracks the ego vehicle
with the spectator chase camera. It does not alter the recorded front-camera
images or schema. Off-screen mode remains the default for final collection and
benchmark runs because it has lower GPU overhead and more stable throughput.

The runner starts the packaged simulator off-screen, loads Town01, collects the
episodes, validates every JSONL row and image reference, writes the manifest,
and shuts the simulator down.

## Exit gate

Phase 2 passes only when:

1. ten episodes complete;
2. exactly 6,000 samples are present;
3. all sample rows satisfy schema version 1.0.0;
4. every referenced JPEG exists;
5. frame IDs are strictly increasing within each episode;
6. episode and route IDs are unique;
7. no staging or failed directory is admitted to the manifest;
8. at least 80 GiB remains free.

Passing the pilot authorizes dataset balancing and expanded expert collection;
it does not yet authorize behavioral-cloning training.
