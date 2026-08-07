#!/usr/bin/env python3
"""Prepare leakage-resistant temporal windows from the accepted Phase 2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.config import load_and_validate_config
from carla_diffusion.dataset import DatasetError, validate_dataset
from carla_diffusion.schema import DrivingSample, validate_sample
from carla_diffusion.splits import split_pilot_routes
from carla_diffusion.training_data import (
    WeightConfig,
    build_temporal_windows,
    normalization_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "phase2_pilot_v2_v043",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase3_pilot_v1",
    )
    return parser.parse_args()


def load_episodes(dataset_root: Path) -> dict[str, list[DrivingSample]]:
    episodes: dict[str, list[DrivingSample]] = {}
    for episode_dir in sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ):
        samples = [
            validate_sample(json.loads(line))
            for line in (episode_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if not samples:
            raise DatasetError(f"empty episode: {episode_dir.name}")
        episodes[episode_dir.name] = samples
    return episodes


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise DatasetError(f"output root already exists: {output_root}")

    structural = validate_dataset(dataset_root)
    if structural["status"] != "passed":
        print(json.dumps(structural, indent=2))
        return 1

    config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    preprocessing = config["preprocessing"]
    episodes = load_episodes(dataset_root)
    route_to_episode: dict[str, str] = {}
    for episode_id, samples in episodes.items():
        route_id = samples[0].route_id
        if route_id in route_to_episode:
            raise DatasetError(f"route occurs in multiple episodes: {route_id}")
        route_to_episode[route_id] = episode_id

    splits = split_pilot_routes(
        route_to_episode,
        seed=int(config["project"]["random_seed"]),
        validation_routes=int(preprocessing["pilot_validation_routes"]),
        test_routes=int(preprocessing["pilot_test_routes"]),
    )
    route_split = {
        route_id: split_name
        for split_name, route_ids in splits.items()
        for route_id in route_ids
    }
    weight_config = WeightConfig(
        stationary_hold=float(preprocessing["stationary_hold_weight"]),
        turning=float(preprocessing["turning_weight"]),
        active_braking=float(preprocessing["active_braking_weight"]),
        lead_context=float(preprocessing["lead_context_weight"]),
    )
    history = int(config["camera"]["history_frames"])
    horizon = int(config["policy"]["action_horizon"])
    acceleration_clip = float(preprocessing["acceleration_clip_mps2"])

    all_windows = []
    windows_by_split: Counter[str] = Counter()
    weight_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    training_samples: list[DrivingSample] = []
    raw_acceleration_outliers = 0
    for samples in episodes.values():
        route_id = samples[0].route_id
        split = route_split[route_id]
        if split == "train":
            training_samples.extend(samples)
        raw_acceleration_outliers += sum(abs(sample.state[1]) > acceleration_clip for sample in samples)
        windows = build_temporal_windows(
            samples,
            split=split,
            history=history,
            horizon=horizon,
            acceleration_clip=acceleration_clip,
            weight_config=weight_config,
        )
        all_windows.extend(windows)
        windows_by_split[split] += len(windows)
        for window in windows:
            weight_counts[f"{window.sample_weight:g}"] += 1
            category_counts.update(window.categories)

    expected_windows = sum(max(0, len(samples) - history - horizon + 2) for samples in episodes.values())
    if len(all_windows) != expected_windows:
        raise DatasetError(
            f"temporal-window mismatch: expected {expected_windows}, got {len(all_windows)}"
        )

    stage_root = output_root.with_name(f".{output_root.name}.inprogress")
    if stage_root.exists():
        raise DatasetError(f"staging root already exists: {stage_root}")
    stage_root.mkdir(parents=True)
    windows_path = stage_root / "windows.jsonl"
    digest = hashlib.sha256()
    with windows_path.open("x", encoding="utf-8") as handle:
        for window in all_windows:
            line = json.dumps(window.to_dict(), separators=(",", ":"), allow_nan=False) + "\n"
            handle.write(line)
            digest.update(line.encode())

    split_payload = {
        "strategy": "deterministic_route_level_single_town_pilot",
        "seed": config["project"]["random_seed"],
        "routes": splits,
        "route_counts": {name: len(values) for name, values in splits.items()},
        "window_counts": dict(sorted(windows_by_split.items())),
    }
    normalization = normalization_statistics(training_samples, acceleration_clip)
    report = {
        "status": "passed",
        "source_dataset": str(dataset_root),
        "output_root": str(output_root),
        "episodes": len(episodes),
        "raw_samples": sum(len(samples) for samples in episodes.values()),
        "temporal_windows": len(all_windows),
        "history_frames": history,
        "action_horizon": horizon,
        "model_state_dimension": preprocessing["model_state_dimension"],
        "categorical_condition_dimension": preprocessing["categorical_condition_dimension"],
        "route_counts": split_payload["route_counts"],
        "window_counts": split_payload["window_counts"],
        "weight_counts": dict(sorted(weight_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "raw_acceleration_outliers_clipped": raw_acceleration_outliers,
        "normalization_training_samples": normalization["training_samples"],
        "windows_sha256": digest.hexdigest(),
    }
    atomic_write_json(stage_root / "splits.json", split_payload)
    atomic_write_json(stage_root / "normalization.json", normalization)
    atomic_write_json(stage_root / "report.json", report)
    os.replace(stage_root, output_root)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
