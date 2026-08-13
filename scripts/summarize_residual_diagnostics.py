#!/usr/bin/env python3
"""Summarize gate timing and residual actions from closed-loop telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = [
        json.loads(line) for line in args.telemetry.read_text().splitlines()
    ]
    active = [row for row in rows if row.get("recovery_active")]
    safety_active = [row for row in rows if row.get("safety_active")]
    preemptive_safety_active = [
        row for row in rows if row.get("preemptive_safety_active")
    ]
    activations = [row for row in rows if row.get("recovery_activated")]
    exits = [row for row in rows if row.get("recovery_exit_reason")]
    liveness_active = [row for row in rows if row.get("liveness_active")]
    liveness_activations = [row for row in rows if row.get("liveness_activated")]
    liveness_exits = [row for row in rows if row.get("liveness_exit_reason")]
    collision = next((row for row in rows if row.get("collision_contact")), None)
    report = {
        "ticks": len(rows),
        "recovery_active_ticks": len(active),
        "safety_active_ticks": len(safety_active),
        "preemptive_safety_active_ticks": len(preemptive_safety_active),
        "liveness_active_ticks": len(liveness_active),
        "liveness_activation_ticks": [row["tick"] for row in liveness_activations],
        "liveness_exits": [
            {"tick": row["tick"], "reason": row["liveness_exit_reason"]}
            for row in liveness_exits
        ],
        "activation_ticks": [row["tick"] for row in activations],
        "recovery_exits": [
            {"tick": row["tick"], "reason": row["recovery_exit_reason"]}
            for row in exits
        ],
        "first_active_tick": active[0]["tick"] if active else None,
        "collision_tick": collision["tick"] if collision else None,
        "ticks_from_first_activation_to_collision": (
            collision["tick"] - active[0]["tick"]
            if collision is not None and active
            else None
        ),
        "maximum_abs_residual_steering": max(
            (abs(float(row.get("residual_steering", 0.0))) for row in active),
            default=0.0,
        ),
        "maximum_abs_residual_longitudinal": max(
            (abs(float(row.get("residual_longitudinal", 0.0))) for row in active),
            default=0.0,
        ),
        "maximum_abs_applied_residual_steering": max(
            (
                abs(float(row.get("applied_residual_steering", 0.0)))
                for row in safety_active
            ),
            default=0.0,
        ),
        "maximum_abs_applied_residual_longitudinal": max(
            (
                abs(float(row.get("applied_residual_longitudinal", 0.0)))
                for row in safety_active
            ),
            default=0.0,
        ),
        "steering_direction_rejection_ticks": sum(
            bool(row.get("steering_direction_rejected")) for row in rows
        ),
        "final_steering_direction_override_ticks": sum(
            bool(row.get("final_steering_direction_overridden")) for row in rows
        ),
        "speed_governor_ticks": sum(
            bool(row.get("speed_governor_active")) for row in rows
        ),
        "deployment_speed_governor_ticks": sum(
            bool(row.get("deployment_speed_governor_active")) for row in rows
        ),
        "deployment_speed_governor_modes": {
            mode: sum(row.get("deployment_speed_governor_mode") == mode for row in rows)
            for mode in ("coast", "brake", "emergency_brake")
        },
        "steering_slew_limited_ticks": sum(
            bool(row.get("steering_slew_limited")) for row in rows
        ),
        "maximum_speed_mps": max(
            (float(row["speed_mps"]) for row in rows), default=0.0
        ),
        "maximum_recovery_speed_mps": max(
            (float(row["speed_mps"]) for row in safety_active), default=0.0
        ),
        "maximum_abs_lane_offset_m": max(
            (abs(float(row["lane_offset_m"])) for row in rows), default=0.0
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
