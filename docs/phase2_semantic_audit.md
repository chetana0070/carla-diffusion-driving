# Phase 2 Semantic Dataset Audit

## Purpose

Schema validity proves that data is readable; it does not prove that the expert
dataset represents the behaviors needed by a driving policy. This audit is the
decision gate between the infrastructure pilot and expanded expert collection.

## Metrics

- episode, sample, simulated-time, and temporal-window yield;
- steering, throttle, coast, and braking distributions;
- active braking separated from stationary brake holding;
- speed, acceleration, lane-offset, and heading-error distributions;
- left, right, straight, and lane-follow command coverage;
- lead-vehicle and traffic-light availability;
- lead-vehicle range integrity, red-light stop duplication, and acceleration outliers;
- collision, lane-invasion, red-light, and intervention labels;
- route-progress distribution;
- JPEG decode, format, and 640 x 360 dimension integrity.

## Outputs

```text
artifacts/evaluations/phase2_semantic_audit.json
artifacts/evaluations/phase2_semantic_audit.md
artifacts/evaluations/phase2_episode_summary.csv
artifacts/evaluations/phase2_contact_sheet.jpg
```

The contact sheet is stratified across turns, braking, hard steering, traffic
lights, and lead-vehicle contexts. It is a mandatory human visual check, not a
replacement for numerical validation.

## Command

```bash
conda activate carla310
cd /home/chetana/carla-diffusion-driving
python scripts/audit_phase2_dataset.py data/raw/phase2_pilot
```

## Decision contract

`ready_to_scale_collection` means the pilot supports a pipeline-scale BC smoke
experiment and expanded balanced collection. It does not mean that ten minutes
of Town01 driving is sufficient for the frozen research comparison.

`targeted_collection_required` means at least one required diversity or quality
gate failed. The failed gates determine the next collection strata; training on
the pilot before correcting them would institutionalize dataset bias.

The collector treats a lead vehicle as valid only when it is ahead on the same
CARLA road and lane and no farther than 80 m. A raw brake command while the ego
vehicle is stopped is reported as a stationary brake hold, not active braking.
