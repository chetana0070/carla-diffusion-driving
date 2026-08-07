# Phase 2.5 Semantic Correction

## Why this correction exists

The first 6,000-sample pilot was structurally valid, but its initial audit gates
were too permissive. Lead-vehicle distances reached 278.6 m because the
collector used only ego-frame lateral proximity. The audit also counted brake
holding at red lights as active braking.

Version 0.4.1 corrects both definitions before expanded collection:

- a lead vehicle must be ahead on the same CARLA road and lane and within 80 m;
- active braking requires longitudinal action below -0.05 while speed is at
  least 0.5 m/s;
- stationary brake holding, red-light stopping, and acceleration outliers are
  reported separately;
- any lead distance beyond 80 m is a required-gate failure.

Version 0.4.3 adds targeted lead-following episodes. Randomly spawned traffic
did not reliably intersect the ego route, so selected episodes now place a
slower vehicle 25--50 m ahead on the same generated route. This is a controlled
coverage stratum, not a replacement for routine traffic episodes.

## Preserve and re-audit the original pilot

Do not delete `data/raw/phase2_pilot`. It is useful evidence that the corrected
gate detects the original semantic defect.

```bash
python scripts/audit_phase2_dataset.py data/raw/phase2_pilot \
  --json-report artifacts/evaluations/phase2_original_reaudit.json \
  --markdown-report artifacts/evaluations/phase2_original_reaudit.md \
  --episode-csv artifacts/evaluations/phase2_original_episode_summary.csv \
  --contact-sheet artifacts/evaluations/phase2_original_contact_sheet.jpg
```

Expected result: `targeted_collection_required`, including a failed
`lead_vehicle_distance_integrity` gate.

## Collect the corrected pilot

Use a new root so the two collector versions cannot be mixed.

```bash
CARLA_RENDER_MODE=live \
PHASE2_EPISODES=10 \
PHASE2_TICKS_PER_EPISODE=600 \
PHASE2_BACKGROUND_VEHICLES=8 \
PHASE2_GUARANTEED_LEAD_EPISODES=4 \
PHASE2_DATASET_ROOT=data/raw/phase2_pilot_v2 \
PHASE2_REPORT_PATH=artifacts/evaluations/phase2_pilot_v2_report.json \
./scripts/run_phase2_pilot.sh
```

Then audit it:

```bash
python scripts/audit_phase2_dataset.py data/raw/phase2_pilot_v2 \
  --json-report artifacts/evaluations/phase2_pilot_v2_semantic_audit.json \
  --markdown-report artifacts/evaluations/phase2_pilot_v2_semantic_audit.md \
  --episode-csv artifacts/evaluations/phase2_pilot_v2_episode_summary.csv \
  --contact-sheet artifacts/evaluations/phase2_pilot_v2_contact_sheet.jpg
```

Expanded multi-town collection remains blocked until the corrected pilot has no
required failures and its contact sheet has been visually inspected.
