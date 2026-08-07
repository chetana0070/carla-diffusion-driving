#!/usr/bin/env python3
"""Audit semantic coverage and visual integrity of the Phase 2 expert dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.audit import (
    counter_report,
    evaluate_readiness,
    fraction,
    numeric_summary,
    usable_temporal_windows,
)
from carla_diffusion.config import load_and_validate_config
from carla_diffusion.dataset import validate_dataset
from carla_diffusion.schema import DrivingSample, validate_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "phase2_pilot",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluations" / "phase2_semantic_audit.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluations" / "phase2_semantic_audit.md",
    )
    parser.add_argument(
        "--episode-csv",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluations" / "phase2_episode_summary.csv",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluations" / "phase2_contact_sheet.jpg",
    )
    return parser.parse_args()


def load_records(
    root: Path,
) -> tuple[list[tuple[DrivingSample, dict[str, Any]]], list[dict[str, Any]]]:
    records: list[tuple[DrivingSample, dict[str, Any]]] = []
    episode_metadata: list[dict[str, Any]] = []
    episode_directories = sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    for episode_dir in episode_directories:
        metadata = json.loads((episode_dir / "episode.json").read_text(encoding="utf-8"))
        episode_metadata.append(metadata)
        for line in (episode_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            records.append((validate_sample(raw), raw))
    return records, episode_metadata


def inspect_images(
    dataset_root: Path,
    records: Iterable[tuple[DrivingSample, dict[str, Any]]],
) -> dict[str, Any]:
    dimensions: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    checked = 0
    for sample, _ in records:
        path = dataset_root / sample.rgb_front_path
        try:
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
                formats[str(image.format)] += 1
                image.verify()
            checked += 1
        except (OSError, UnidentifiedImageError) as error:
            failures.append({"path": sample.rgb_front_path, "error": str(error)})
    return {
        "checked": checked,
        "failures": failures,
        "failure_count": len(failures),
        "dimensions": dict(sorted(dimensions.items())),
        "formats": dict(sorted(formats.items())),
    }


def select_spread(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(items) <= count:
        return list(items)
    indices = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[index] for index in indices]


def build_contact_sheet(
    dataset_root: Path,
    candidates: dict[str, list[dict[str, Any]]],
    fallback: list[dict[str, Any]],
    output: Path,
) -> int:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    priorities = [
        ("left", 3),
        ("right", 3),
        ("straight", 3),
        ("braking", 3),
        ("hard_steering", 3),
        ("traffic_light", 3),
        ("lead_vehicle", 3),
    ]
    for category, count in priorities:
        for item in select_spread(candidates.get(category, []), count):
            if item["image_path"] not in seen:
                selected.append(item)
                seen.add(item["image_path"])
    for item in select_spread(fallback, 24):
        if len(selected) >= 24:
            break
        if item["image_path"] not in seen:
            selected.append(item)
            seen.add(item["image_path"])

    columns = 4
    tile_width, image_height, label_height = 320, 180, 42
    rows = (len(selected) + columns - 1) // columns
    sheet_size = (columns * tile_width, rows * (image_height + label_height))
    sheet = Image.new("RGB", sheet_size, "#111827")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(selected):
        row, column = divmod(index, columns)
        x, y = column * tile_width, row * (image_height + label_height)
        with Image.open(dataset_root / item["image_path"]) as source:
            tile = ImageOps.fit(source.convert("RGB"), (tile_width, image_height))
        sheet.paste(tile, (x, y))
        draw.text((x + 6, y + image_height + 4), item["label_line_1"], fill="white")
        draw.text((x + 6, y + image_height + 21), item["label_line_2"], fill="#93c5fd")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=90)
    return len(selected)


def episode_rows(
    records: list[tuple[DrivingSample, dict[str, Any]]],
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[DrivingSample]] = defaultdict(list)
    for sample, _ in records:
        grouped[sample.episode_id].append(sample)
    metadata_by_id = {item["episode_id"]: item for item in metadata}
    rows: list[dict[str, Any]] = []
    for episode_id, samples in sorted(grouped.items()):
        command_counts = Counter(sample.route_command for sample in samples)
        rows.append(
            {
                "episode_id": episode_id,
                "route_id": samples[0].route_id,
                "samples": len(samples),
                "mean_speed_mps": sum(sample.state[0] for sample in samples) / len(samples),
                "braking_samples": sum(sample.expert_action[1] < -0.05 for sample in samples),
                "turn_samples": sum(abs(sample.expert_action[0]) >= 0.05 for sample in samples),
                "left_commands": command_counts["left"],
                "right_commands": command_counts["right"],
                "straight_commands": command_counts["straight"],
                "collisions": sum(sample.collision for sample in samples),
                "lane_invasions": sum(sample.lane_invasion for sample in samples),
                "route_progress_fraction": metadata_by_id[episode_id].get(
                    "route_progress_fraction"
                ),
            }
        )
    return rows


def write_episode_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_metrics(
    records: list[tuple[DrivingSample, dict[str, Any]]],
    metadata: list[dict[str, Any]],
    config: dict[str, Any],
    image_integrity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    samples = [sample for sample, _ in records]
    total = len(samples)
    steering = [sample.expert_action[0] for sample in samples]
    longitudinal = [sample.expert_action[1] for sample in samples]
    speeds = [sample.state[0] for sample in samples]
    accelerations = [sample.state[1] for sample in samples]
    lane_offsets = [sample.state[3] for sample in samples]
    heading_errors = [sample.state[4] for sample in samples]
    lead_distances = [sample.state[5] for sample in samples if sample.state[5] >= 0]
    light_distances = [sample.state[7] for sample in samples if sample.state[7] >= 0]
    active_braking = sum(
        action < -0.05 and speed >= 0.5
        for action, speed in zip(longitudinal, speeds, strict=True)
    )
    stationary_brake_holds = sum(
        action < -0.05 and speed < 0.5
        for action, speed in zip(longitudinal, speeds, strict=True)
    )
    red_light_stops = sum(
        sample.traffic_light_state == "red" and sample.state[0] < 0.5
        for sample in samples
    )
    acceleration_outliers = sum(abs(value) > 12.0 for value in accelerations)
    lead_distance_outliers = sum(value > 80.0 for value in lead_distances)
    commands = Counter(sample.route_command for sample in samples)
    lights = Counter(sample.traffic_light_state for sample in samples)
    event_counts = {
        "collision": sum(sample.collision for sample in samples),
        "lane_invasion": sum(sample.lane_invasion for sample in samples),
        "red_light_violation": sum(sample.red_light_violation for sample in samples),
        "intervention": sum(sample.intervention for sample in samples),
    }
    episode_counts = Counter(sample.episode_id for sample in samples)
    metrics: dict[str, Any] = {
        "episodes": len(episode_counts),
        "samples": total,
        "simulated_minutes": total * config["simulator"]["fixed_delta_seconds"] / 60.0,
        "usable_temporal_windows": usable_temporal_windows(
            episode_counts.values(),
            config["camera"]["history_frames"],
            config["policy"]["action_horizon"],
        ),
        "actions": {
            "steering": numeric_summary(steering),
            "absolute_steering": numeric_summary([abs(value) for value in steering]),
            "longitudinal": numeric_summary(longitudinal),
            "near_straight_fraction": fraction(sum(abs(value) < 0.05 for value in steering), total),
            "turning_fraction": fraction(sum(abs(value) >= 0.05 for value in steering), total),
            "hard_steering_fraction": fraction(
                sum(abs(value) >= 0.25 for value in steering), total
            ),
            "braking_fraction": fraction(sum(value < -0.05 for value in longitudinal), total),
            "active_braking_fraction": fraction(active_braking, total),
            "stationary_brake_hold_fraction": fraction(stationary_brake_holds, total),
            "coasting_fraction": fraction(sum(abs(value) <= 0.05 for value in longitudinal), total),
            "throttle_fraction": fraction(sum(value > 0.05 for value in longitudinal), total),
        },
        "states": {
            "speed_mps": numeric_summary(speeds),
            "longitudinal_acceleration_mps2": numeric_summary(accelerations),
            "absolute_lane_offset_m": numeric_summary([abs(value) for value in lane_offsets]),
            "absolute_heading_error_rad": numeric_summary([abs(value) for value in heading_errors]),
            "stopped_fraction": fraction(sum(value < 0.5 for value in speeds), total),
            "moving_fraction": fraction(sum(value >= 0.5 for value in speeds), total),
            "lead_vehicle_available_fraction": fraction(len(lead_distances), total),
            "lead_vehicle_distance_m": numeric_summary(lead_distances),
            "traffic_light_available_fraction": fraction(len(light_distances), total),
            "traffic_light_distance_m": numeric_summary(light_distances),
            "red_light_stopped_fraction": fraction(red_light_stops, total),
            "acceleration_outlier_fraction": fraction(acceleration_outliers, total),
            "lead_vehicle_distance_outlier_count": lead_distance_outliers,
        },
        "categorical": {
            "route_command": counter_report(commands, total),
            "traffic_light_state": counter_report(lights, total),
        },
        "events": {
            name: {"count": count, "fraction": fraction(count, total)}
            for name, count in event_counts.items()
        },
        "route_progress_fraction": numeric_summary(
            [float(item.get("route_progress_fraction", 0.0)) for item in metadata]
        ),
        "image_integrity": image_integrity,
    }

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fallback: list[dict[str, Any]] = []
    for sample in samples:
        item = {
            "image_path": sample.rgb_front_path,
            "label_line_1": (
                f"{sample.route_command} | {sample.traffic_light_state} | "
                f"v={sample.state[0]:.1f}m/s"
            ),
            "label_line_2": (
                f"steer={sample.expert_action[0]:+.2f} "
                f"long={sample.expert_action[1]:+.2f}"
            ),
        }
        fallback.append(item)
        candidates[sample.route_command].append(item)
        if sample.expert_action[1] < -0.05:
            candidates["braking"].append(item)
        if abs(sample.expert_action[0]) >= 0.25:
            candidates["hard_steering"].append(item)
        if sample.traffic_light_state in {"red", "yellow", "green"}:
            candidates["traffic_light"].append(item)
        if sample.state[5] >= 0:
            candidates["lead_vehicle"].append(item)
    return metrics, candidates, fallback


def write_markdown(report: dict[str, Any], output: Path) -> None:
    metrics = report["metrics"]
    lines = [
        "# Phase 2 Semantic Dataset Audit",
        "",
        f"**Decision:** `{report['decision']}`",
        "",
        f"- Episodes: {metrics['episodes']}",
        f"- Samples: {metrics['samples']}",
        f"- Simulated driving: {metrics['simulated_minutes']:.2f} minutes",
        f"- Usable temporal windows: {metrics['usable_temporal_windows']}",
        f"- Contact-sheet frames: {report['contact_sheet_frames']}",
        "",
        "## Readiness gates",
        "",
        "| Gate | Status | Observed | Threshold | Required |",
        "|---|---|---|---|---|",
    ]
    for gate in report["gates"]:
        lines.append(
            f"| {gate['name']} | {gate['status']} | `{gate['observed']}` | "
            f"{gate['threshold']} | {gate['required']} |"
        )
    lines.extend(
        [
            "",
            "## Key coverage",
            "",
            f"- Turning fraction: {metrics['actions']['turning_fraction']:.2%}",
            f"- Raw braking fraction: {metrics['actions']['braking_fraction']:.2%}",
            f"- Active braking fraction: {metrics['actions']['active_braking_fraction']:.2%}",
            (
                "- Stationary brake-hold fraction: "
                f"{metrics['actions']['stationary_brake_hold_fraction']:.2%}"
            ),
            f"- Moving fraction: {metrics['states']['moving_fraction']:.2%}",
            (
                "- Red-light stopped fraction: "
                f"{metrics['states']['red_light_stopped_fraction']:.2%}"
            ),
            (
                "- Acceleration outlier fraction: "
                f"{metrics['states']['acceleration_outlier_fraction']:.2%}"
            ),
            (
                "- Lead-vehicle availability: "
                f"{metrics['states']['lead_vehicle_available_fraction']:.2%}"
            ),
            (
                "- Traffic-light availability: "
                f"{metrics['states']['traffic_light_available_fraction']:.2%}"
            ),
            f"- Corrupt images: {metrics['image_integrity']['failure_count']}",
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    structural = validate_dataset(dataset_root)
    if structural["status"] != "passed":
        print(json.dumps(structural, indent=2))
        return 1

    config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    records, metadata = load_records(dataset_root)
    image_integrity = inspect_images(dataset_root, records)
    metrics, candidates, fallback = build_metrics(records, metadata, config, image_integrity)
    gates = evaluate_readiness(metrics)
    required_failures = [
        gate["name"]
        for gate in gates
        if gate["required"] and gate["status"] == "failed"
    ]
    advisory_failures = [
        gate["name"]
        for gate in gates
        if not gate["required"] and gate["status"] == "failed"
    ]
    decision = "targeted_collection_required" if required_failures else "ready_to_scale_collection"
    if required_failures:
        interpretation = (
            "The pilot is structurally valid but is not behaviorally balanced "
            "enough to scale as-is. "
            "Collect targeted episodes for the failed required gates before training."
        )
    else:
        interpretation = (
            "The pilot has sufficient control diversity for a pipeline-scale BC smoke experiment. "
            "It remains too small for the frozen research comparison; scale collection across the "
            "configured train towns, weather, traffic, and rare-event strata."
        )

    contact_count = build_contact_sheet(dataset_root, candidates, fallback, args.contact_sheet)
    rows = episode_rows(records, metadata)
    write_episode_csv(rows, args.episode_csv)
    report = {
        "status": "passed",
        "decision": decision,
        "required_failures": required_failures,
        "advisory_failures": advisory_failures,
        "dataset_root": str(dataset_root),
        "structural_validation": structural,
        "metrics": metrics,
        "gates": gates,
        "contact_sheet": str(args.contact_sheet.resolve()),
        "contact_sheet_frames": contact_count,
        "episode_csv": str(args.episode_csv.resolve()),
        "interpretation": interpretation,
    }
    args.json_report.parent.mkdir(parents=True, exist_ok=True)
    args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.markdown_report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
