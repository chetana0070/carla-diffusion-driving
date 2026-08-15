#!/usr/bin/env python3
"""Convert frozen temporal windows into language-conditioned VLA examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.config import load_and_validate_config
from carla_diffusion.vla_data import build_vla_record, build_vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase3_pilot_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase8_vla_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise RuntimeError(f"output root already exists: {output_root}; use --overwrite")
        shutil.rmtree(output_root)
    config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    vla = config["vision_language_action"]
    source_report = json.loads((input_root / "report.json").read_text(encoding="utf-8"))
    source_windows = [
        json.loads(line)
        for line in (input_root / "windows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vocabulary = build_vocabulary()
    records = [
        build_vla_record(
            window,
            vocabulary=vocabulary,
            planner_horizon=int(vla["action_horizon"]),
            max_instruction_tokens=int(vla["max_instruction_tokens"]),
        )
        for window in source_windows
    ]
    routes_by_split: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    for record in records:
        routes_by_split[record.split].add(record.route_id)
        counts[record.split] += 1
        language_counts[f"{record.route_command}:{record.traffic_light_state}"] += 1
    splits = tuple(sorted(routes_by_split))
    for index, split in enumerate(splits):
        for other in splits[index + 1 :]:
            overlap = routes_by_split[split] & routes_by_split[other]
            if overlap:
                raise RuntimeError(f"route leakage between {split} and {other}: {sorted(overlap)}")

    stage = output_root.with_name(f".{output_root.name}.inprogress")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    digest = hashlib.sha256()
    with (stage / "vla_windows.jsonl").open("x", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record.to_dict(), separators=(",", ":"), allow_nan=False) + "\n"
            handle.write(line)
            digest.update(line.encode())
    shutil.copy2(input_root / "normalization.json", stage / "normalization.json")
    atomic_json(stage / "vocabulary.json", vocabulary)
    report = {
        "status": "passed",
        "contract_version": "1.0.0",
        "source_processed_root": str(input_root),
        "source_dataset": source_report["source_dataset"],
        "output_root": str(output_root),
        "records": len(records),
        "split_counts": dict(sorted(counts.items())),
        "route_counts": {
            split: len(routes) for split, routes in sorted(routes_by_split.items())
        },
        "language_condition_counts": dict(sorted(language_counts.items())),
        "vocabulary_size": len(vocabulary),
        "history_frames": vla["history_frames"],
        "action_horizon": vla["action_horizon"],
        "execute_steps": vla["execute_steps"],
        "planner_hz": vla["planner_hz"],
        "control_hz": config["simulator"]["control_hz"],
        "vla_windows_sha256": digest.hexdigest(),
        "claim_boundary": (
            "Instructions are deterministic labels derived from existing route and signal "
            "metadata; this artifact does not demonstrate open-vocabulary grounding."
        ),
    }
    atomic_json(stage / "report.json", report)
    os.replace(stage, output_root)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
