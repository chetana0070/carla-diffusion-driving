#!/usr/bin/env python3
"""Audit corrective episodes before they enter retraining."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.config import load_and_validate_config
from carla_diffusion.dataset import validate_dataset
from carla_diffusion.schema import validate_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "phase6_corrections_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluations" / "phase6_audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve()
    config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    structural = validate_dataset(root)
    issues = list(structural.get("issues", []))
    reason_counts: Counter[str] = Counter()
    samples = 0
    intervention_samples = 0
    collision_samples = 0
    lane_invasion_samples = 0
    episode_rows: list[dict[str, Any]] = []
    if structural["status"] == "passed":
        for episode_dir in sorted(
            path
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ):
            metadata = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
            local_samples = 0
            local_interventions = 0
            for line in (episode_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines():
                sample = validate_sample(json.loads(line))
                samples += 1
                local_samples += 1
                intervention_samples += int(sample.intervention)
                local_interventions += int(sample.intervention)
                collision_samples += int(sample.collision)
                lane_invasion_samples += int(sample.lane_invasion)
            reasons = [str(reason) for reason in metadata.get("intervention_reasons", [])]
            reason_counts.update(reasons)
            if local_interventions != local_samples:
                issues.append(f"{episode_dir.name}: contains non-intervention samples")
            if local_samples < int(config["corrective_collection"]["minimum_published_samples"]):
                issues.append(f"{episode_dir.name}: insufficient temporal-window support")
            episode_rows.append(
                {
                    "episode_id": episode_dir.name,
                    "samples": local_samples,
                    "intervention_reasons": reasons,
                }
            )
    full_gate = (
        structural.get("episodes", 0)
        >= int(config["corrective_collection"]["minimum_intervention_episodes"])
        and not issues
        and collision_samples == 0
    )
    report = {
        "status": "passed" if structural["status"] == "passed" and not issues else "failed",
        "dataset_root": str(root),
        "episodes": structural.get("episodes", 0),
        "samples": samples,
        "intervention_samples": intervention_samples,
        "collision_samples": collision_samples,
        "lane_invasion_samples": lane_invasion_samples,
        "intervention_reason_counts": dict(sorted(reason_counts.items())),
        "minimum_intervention_episodes": config["corrective_collection"][
            "minimum_intervention_episodes"
        ],
        "full_collection_gate_passed": full_gate,
        "episode_reports": episode_rows,
        "issues": issues,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
