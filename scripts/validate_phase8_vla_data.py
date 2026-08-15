#!/usr/bin/env python3
"""Validate the frozen Phase 8 VLA data contract and digest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "processed_root",
        nargs="?",
        type=Path,
        default=Path("data/processed/phase8_vla_v1"),
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def validate(root: Path) -> dict[str, Any]:
    issues: list[str] = []
    required = (
        "report.json",
        "normalization.json",
        "vocabulary.json",
        "vla_windows.jsonl",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        return {"status": "failed", "processed_root": str(root), "issues": missing}
    source_report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((root / "vocabulary.json").read_text(encoding="utf-8"))
    lines = (root / "vla_windows.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
    digest = hashlib.sha256("".join(lines).encode()).hexdigest()
    if digest != source_report["vla_windows_sha256"]:
        issues.append("VLA window digest mismatch")
    routes: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    instruction_lengths: Counter[int] = Counter()
    horizon: int | None = None
    for line_number, line in enumerate(lines, start=1):
        row = json.loads(line)
        split = str(row.get("split"))
        if split not in {"train", "validation", "test"}:
            issues.append(f"line {line_number}: invalid split")
        routes[split].add(str(row.get("route_id")))
        counts[split] += 1
        token_ids = row.get("instruction_token_ids", [])
        mask = row.get("instruction_attention_mask", [])
        if not token_ids or len(token_ids) != len(mask):
            issues.append(f"line {line_number}: invalid instruction encoding")
        if any(token not in vocabulary.values() for token in token_ids):
            issues.append(f"line {line_number}: token ID is outside vocabulary")
        instruction_lengths[len(token_ids)] += 1
        actions = row.get("action_chunk_target", [])
        horizon = len(actions) if horizon is None else horizon
        if len(actions) != horizon:
            issues.append(f"line {line_number}: inconsistent action horizon")
        if any(
            len(action) != 2
            or any(not math.isfinite(float(value)) or abs(float(value)) > 1 for value in action)
            for action in actions
        ):
            issues.append(f"line {line_number}: invalid bounded action target")
    split_names = tuple(sorted(routes))
    for index, split in enumerate(split_names):
        for other in split_names[index + 1 :]:
            if routes[split] & routes[other]:
                issues.append(f"route leakage between {split} and {other}")
    return {
        "status": "passed" if not issues else "failed",
        "processed_root": str(root),
        "records": sum(counts.values()),
        "split_counts": dict(sorted(counts.items())),
        "route_counts": {name: len(values) for name, values in sorted(routes.items())},
        "action_horizon": horizon,
        "instruction_width_counts": {
            str(width): count for width, count in sorted(instruction_lengths.items())
        },
        "digest_verified": digest == source_report["vla_windows_sha256"],
        "issues": issues,
    }


def main() -> int:
    args = parse_args()
    payload = validate(args.processed_root.resolve())
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
