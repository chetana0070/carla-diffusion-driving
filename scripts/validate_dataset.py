#!/usr/bin/env python3
"""Validate a collected CARLA dataset against the frozen sample contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.dataset import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "phase2_pilot",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = validate_dataset(args.dataset_root)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
