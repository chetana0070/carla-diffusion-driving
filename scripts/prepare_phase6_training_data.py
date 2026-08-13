#!/usr/bin/env python3
"""Merge corrective demonstrations into training without changing evaluation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.config import load_and_validate_config
from carla_diffusion.dataset import DatasetError, validate_dataset
from carla_diffusion.schema import DrivingSample, validate_sample
from carla_diffusion.training_data import WeightConfig, build_temporal_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase3_pilot_v1",
    )
    parser.add_argument(
        "--correction-dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "phase6_corrections_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase6_corrective_v2",
    )
    parser.add_argument("--correction-weight-multiplier", type=float, default=1.5)
    parser.add_argument("--correction-validation-episodes", type=int, default=2)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_correction_episodes(root: Path) -> dict[str, list[DrivingSample]]:
    episodes: dict[str, list[DrivingSample]] = {}
    for episode_dir in sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    ):
        samples = [
            validate_sample(json.loads(line))
            for line in (episode_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if not samples:
            raise DatasetError(f"empty correction episode: {episode_dir.name}")
        if not all(sample.intervention for sample in samples):
            raise DatasetError(f"non-intervention sample in correction episode: {episode_dir.name}")
        if any(sample.collision for sample in samples):
            raise DatasetError(f"collision sample in correction episode: {episode_dir.name}")
        episodes[episode_dir.name] = samples
    return episodes


def rows_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, separators=(",", ":"), allow_nan=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def correction_validation_ids(
    episode_ids: list[str], *, count: int, seed: int
) -> set[str]:
    if count < 1 or count >= len(episode_ids):
        raise ValueError("correction validation count must leave training episodes")

    def rank(episode_id: str) -> str:
        return hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()

    return set(sorted(episode_ids, key=rank)[:count])


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    base_root = args.base_processed_root.resolve()
    correction_root = args.correction_dataset_root.resolve()
    output_root = args.output_root.resolve()
    if args.correction_weight_multiplier <= 0:
        raise ValueError("correction weight multiplier must be positive")
    if output_root.exists():
        raise DatasetError(f"output root already exists: {output_root}")

    correction_validation = validate_dataset(correction_root)
    if correction_validation["status"] != "passed":
        print(json.dumps(correction_validation, indent=2))
        return 1

    config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    base_report = load_json(base_root / "report.json")
    base_splits = load_json(base_root / "splits.json")
    base_rows = load_rows(base_root / "windows.jsonl")
    correction_episodes = load_correction_episodes(correction_root)
    base_routes = {
        split: {str(route) for route in base_splits["routes"][split]}
        for split in ("train", "validation", "test")
    }
    if any(
        base_routes[left] & base_routes[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise DatasetError("base route splits overlap")

    correction_routes = {samples[0].route_id for samples in correction_episodes.values()}
    if correction_routes & set().union(*base_routes.values()):
        raise DatasetError("correction route collides with a frozen base route")

    preprocessing = config["preprocessing"]
    weight_config = WeightConfig(
        stationary_hold=float(preprocessing["stationary_hold_weight"]),
        turning=float(preprocessing["turning_weight"]),
        active_braking=float(preprocessing["active_braking_weight"]),
        lead_context=float(preprocessing["lead_context_weight"]),
    )
    history = int(config["camera"]["history_frames"])
    horizon = int(config["policy"]["action_horizon"])
    acceleration_clip = float(preprocessing["acceleration_clip_mps2"])
    correction_rows: list[dict[str, Any]] = []
    correction_training_rows: list[dict[str, Any]] = []
    correction_validation_rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    held_out_episode_ids = correction_validation_ids(
        list(correction_episodes),
        count=args.correction_validation_episodes,
        seed=int(config["project"]["random_seed"]),
    )
    correction_training_routes: set[str] = set()
    correction_validation_routes: set[str] = set()
    for episode_id, samples in correction_episodes.items():
        split = (
            "correction_validation"
            if episode_id in held_out_episode_ids
            else "train"
        )
        windows = build_temporal_windows(
            samples,
            split=split,
            history=history,
            horizon=horizon,
            acceleration_clip=acceleration_clip,
            weight_config=weight_config,
            source_dataset=str(correction_root),
        )
        for window in windows:
            multiplier = (
                args.correction_weight_multiplier if split == "train" else 1.0
            )
            weighted = replace(
                window,
                sample_weight=window.sample_weight * multiplier,
                categories=window.categories + ("corrective_intervention",),
            )
            row = weighted.to_dict()
            correction_rows.append(row)
            if split == "train":
                correction_training_rows.append(row)
                correction_training_routes.add(samples[0].route_id)
            else:
                correction_validation_rows.append(row)
                correction_validation_routes.add(samples[0].route_id)
            category_counts.update(weighted.categories)

    expected = sum(
        max(0, len(samples) - history - horizon + 2)
        for samples in correction_episodes.values()
    )
    if len(correction_rows) != expected:
        raise DatasetError(
            f"correction-window mismatch: expected {expected}, got {len(correction_rows)}"
        )

    frozen_rows = [row for row in base_rows if row["split"] in {"validation", "test"}]
    frozen_digest = rows_digest(frozen_rows)
    merged_rows = base_rows + correction_rows
    merged_evaluation_rows = [
        row for row in merged_rows if row["split"] in {"validation", "test"}
    ]
    if rows_digest(merged_evaluation_rows) != frozen_digest:
        raise DatasetError("frozen validation/test rows changed during merge")

    stage_root = output_root.with_name(f".{output_root.name}.inprogress")
    if stage_root.exists():
        raise DatasetError(f"staging root already exists: {stage_root}")
    stage_root.mkdir(parents=True)
    digest = hashlib.sha256()
    with (stage_root / "windows.jsonl").open("x", encoding="utf-8") as handle:
        for row in merged_rows:
            line = json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
            handle.write(line)
            digest.update(line.encode())

    merged_routes = {
        "train": sorted(base_routes["train"] | correction_training_routes),
        "validation": sorted(base_routes["validation"]),
        "test": sorted(base_routes["test"]),
        "correction_validation": sorted(correction_validation_routes),
    }
    window_counts = Counter(str(row["split"]) for row in merged_rows)
    split_payload = {
        **base_splits,
        "strategy": "frozen_base_splits_plus_episode_level_corrective_holdout",
        "routes": merged_routes,
        "route_counts": {name: len(routes) for name, routes in merged_routes.items()},
        "window_counts": dict(sorted(window_counts.items())),
        "frozen_evaluation_rows_sha256": frozen_digest,
    }
    shutil.copyfile(base_root / "normalization.json", stage_root / "normalization.json")
    atomic_write_json(stage_root / "splits.json", split_payload)
    report = {
        "status": "passed",
        "source_dataset": str(Path(str(base_report["source_dataset"])).resolve()),
        "base_processed_root": str(base_root),
        "correction_dataset": str(correction_root),
        "output_root": str(output_root),
        "base_windows": len(base_rows),
        "correction_episodes": len(correction_episodes),
        "correction_samples": sum(len(samples) for samples in correction_episodes.values()),
        "correction_windows": len(correction_rows),
        "correction_training_episodes": len(correction_episodes) - len(held_out_episode_ids),
        "correction_training_samples": sum(
            len(samples)
            for episode_id, samples in correction_episodes.items()
            if episode_id not in held_out_episode_ids
        ),
        "correction_training_windows": len(correction_training_rows),
        "correction_validation_episodes": len(held_out_episode_ids),
        "correction_validation_samples": sum(
            len(samples)
            for episode_id, samples in correction_episodes.items()
            if episode_id in held_out_episode_ids
        ),
        "correction_validation_episode_ids": sorted(held_out_episode_ids),
        "correction_validation_windows": len(correction_validation_rows),
        "correction_weight_multiplier": args.correction_weight_multiplier,
        "merged_windows": len(merged_rows),
        "window_counts": dict(sorted(window_counts.items())),
        "correction_category_counts": dict(sorted(category_counts.items())),
        "normalization_policy": "copied_from_frozen_base_training_split",
        "validation_test_rows_unchanged": True,
        "frozen_evaluation_rows_sha256": frozen_digest,
        "windows_sha256": digest.hexdigest(),
    }
    atomic_write_json(stage_root / "report.json", report)
    os.replace(stage_root, output_root)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
