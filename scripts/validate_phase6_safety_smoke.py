#!/usr/bin/env python3
"""Validate the controlled Phase 6.4 same-seed safety smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/evaluations/phase6_safety_smoke_v142.json"),
    )
    parser.add_argument(
        "--telemetry-dir",
        type=Path,
        default=Path("artifacts/evaluations/phase6_safety_smoke_v142_telemetry"),
    )
    parser.add_argument("--minimum-ticks", type=int, default=300)
    parser.add_argument("--maximum-recovery-speed-mps", type=float, default=5.0)
    parser.add_argument("--maximum-lane-offset-m", type=float, default=1.5)
    parser.add_argument("--maximum-lane-invasions", type=int, default=1)
    parser.add_argument("--maximum-latency-ms", type=float, default=100.0)
    parser.add_argument("--cold-start-ticks", type=int, default=1)
    parser.add_argument("--maximum-cold-start-latency-ms", type=float, default=200.0)
    return parser.parse_args()


def evaluate_acceptance(
    report: dict[str, Any],
    telemetry_rows: list[dict[str, Any]],
    *,
    minimum_ticks: int,
    maximum_recovery_speed_mps: float,
    maximum_lane_offset_m: float,
    maximum_lane_invasions: int,
    maximum_latency_ms: float,
    cold_start_ticks: int = 1,
    maximum_cold_start_latency_ms: float = 200.0,
) -> dict[str, Any]:
    episodes = report.get("episodes", [])
    if len(episodes) != 1:
        return {
            "status": "failed",
            "gate_passed": False,
            "failures": ["exactly_one_episode_required"],
        }
    episode = episodes[0]
    recovery_exits = [
        str(row["recovery_exit_reason"])
        for row in telemetry_rows
        if row.get("recovery_exit_reason")
    ]
    telemetry_has_recovery_state = bool(telemetry_rows) and all(
        "recovery_active" in row and "recovery_activated" in row
        for row in telemetry_rows
    )
    recovery_activations = sum(
        bool(row.get("recovery_activated")) for row in telemetry_rows
    )
    recovery_active_at_end = bool(
        telemetry_rows and telemetry_rows[-1].get("recovery_active")
    )
    if telemetry_has_recovery_state:
        recovery_disengaged = (
            not recovery_active_at_end
            and len(recovery_exits) >= recovery_activations
        )
    else:
        # Compatibility for telemetry produced before recovery state was recorded.
        recovery_disengaged = bool(recovery_exits)
    telemetry_has_liveness_state = bool(telemetry_rows) and all(
        "liveness_active" in row and "liveness_activated" in row
        for row in telemetry_rows
    )
    liveness_activations = sum(
        bool(row.get("liveness_activated")) for row in telemetry_rows
    )
    liveness_exits = [
        str(row["liveness_exit_reason"])
        for row in telemetry_rows
        if row.get("liveness_exit_reason")
    ]
    liveness_active_at_end = bool(
        telemetry_rows and telemetry_rows[-1].get("liveness_active")
    )
    liveness_disengaged = (
        not telemetry_has_liveness_state
        or (
            not liveness_active_at_end
            and len(liveness_exits) >= liveness_activations
        )
    )
    maximum_recovery_speed = max(
        (float(row["speed_mps"]) for row in telemetry_rows), default=0.0
    )
    maximum_offset = max(
        (abs(float(row["lane_offset_m"])) for row in telemetry_rows), default=0.0
    )
    latency_rows = [
        (int(row["tick"]), float(row["policy_pipeline_latency_ms"]))
        for row in telemetry_rows
        if isinstance(row.get("policy_pipeline_latency_ms"), (int, float))
    ]
    cold_start_latencies = [
        value for tick, value in latency_rows if tick < cold_start_ticks
    ]
    steady_state_latencies = [
        value for tick, value in latency_rows if tick >= cold_start_ticks
    ]
    maximum_cold_start_latency = max(cold_start_latencies, default=float("inf"))
    maximum_steady_state_latency = max(steady_state_latencies, default=float("inf"))
    checks = {
        "report_status": report.get("status") == "passed",
        "tick_completion": int(episode.get("ticks_completed", 0)) >= minimum_ticks,
        "collision_free": int(episode.get("collision_events", 0)) == 0,
        "recovery_speed": maximum_recovery_speed <= maximum_recovery_speed_mps,
        "lane_offset": maximum_offset < maximum_lane_offset_m,
        "lane_invasions": (
            int(episode.get("lane_invasion_events", maximum_lane_invasions + 1))
            <= maximum_lane_invasions
        ),
        "recovery_disengaged": recovery_disengaged,
        "liveness_disengaged": liveness_disengaged,
        "cold_start_latency": (
            maximum_cold_start_latency <= maximum_cold_start_latency_ms
        ),
        "steady_state_latency": maximum_steady_state_latency <= maximum_latency_ms,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failures else "failed",
        "gate_passed": not failures,
        "checks": checks,
        "failures": failures,
        "metrics": {
            "ticks_completed": int(episode.get("ticks_completed", 0)),
            "collision_events": int(episode.get("collision_events", 0)),
            "lane_invasion_events": int(episode.get("lane_invasion_events", 0)),
            "maximum_lane_offset_m": maximum_offset,
            "maximum_recovery_speed_mps": maximum_recovery_speed,
            "maximum_rollout_speed_mps": maximum_recovery_speed,
            "cold_start_ticks": cold_start_ticks,
            "maximum_cold_start_policy_latency_ms": maximum_cold_start_latency,
            "maximum_steady_state_policy_latency_ms": maximum_steady_state_latency,
            "recovery_exit_reasons": recovery_exits,
            "recovery_activation_events": recovery_activations,
            "recovery_active_at_end": recovery_active_at_end,
            "liveness_activation_events": liveness_activations,
            "liveness_exit_reasons": liveness_exits,
            "liveness_active_at_end": liveness_active_at_end,
        },
    }


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    telemetry_files = sorted(args.telemetry_dir.glob("*.jsonl"))
    if len(telemetry_files) != 1:
        raise SystemExit("expected exactly one Phase 6.4 telemetry JSONL file")
    telemetry_rows = [
        json.loads(line)
        for line in telemetry_files[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = evaluate_acceptance(
        report,
        telemetry_rows,
        minimum_ticks=args.minimum_ticks,
        maximum_recovery_speed_mps=args.maximum_recovery_speed_mps,
        maximum_lane_offset_m=args.maximum_lane_offset_m,
        maximum_lane_invasions=args.maximum_lane_invasions,
        maximum_latency_ms=args.maximum_latency_ms,
        cold_start_ticks=args.cold_start_ticks,
        maximum_cold_start_latency_ms=args.maximum_cold_start_latency_ms,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
