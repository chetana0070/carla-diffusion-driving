#!/usr/bin/env python3
"""Validate the frozen Phase 7.5 three-seed closed-loop contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.config import load_and_validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/evaluations/phase7_factorized_closed_loop_v210.json"),
    )
    parser.add_argument(
        "--telemetry-dir",
        type=Path,
        default=Path(
            "artifacts/evaluations/phase7_factorized_closed_loop_v210_telemetry"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/evaluations/phase7_factorized_closed_loop_v210_acceptance.json"
        ),
    )
    return parser.parse_args()


def evaluate_acceptance(
    report: dict[str, Any],
    telemetry_by_seed: dict[int, list[dict[str, Any]]],
    *,
    expected_seeds: list[int],
    expected_checkpoint_sha256: str,
    minimum_ticks: int,
    minimum_mean_route_progress_fraction: float,
    minimum_mean_distance_m: float,
    maximum_collisions: int,
    maximum_lane_invasions: int,
    maximum_red_light_violations: int,
    maximum_no_progress_terminations: int,
    maximum_stationary_fraction: float,
    maximum_speed_mps: float,
    maximum_lane_offset_m: float,
    cold_start_ticks: int,
    maximum_cold_start_latency_ms: float,
    maximum_steady_state_latency_ms: float,
) -> dict[str, Any]:
    episodes = report.get("episodes", [])
    observed_seeds = [int(episode.get("seed", -1)) for episode in episodes]
    aggregate = report.get("aggregate", {})
    telemetry_complete = set(telemetry_by_seed) == set(expected_seeds) and all(
        telemetry_by_seed.get(seed) for seed in expected_seeds
    )
    all_rows = [
        row
        for seed in expected_seeds
        for row in telemetry_by_seed.get(seed, [])
    ]
    cold_latencies = [
        float(row["policy_pipeline_latency_ms"])
        for row in all_rows
        if int(row.get("tick", -1)) < cold_start_ticks
        and isinstance(row.get("policy_pipeline_latency_ms"), (int, float))
    ]
    steady_latencies = [
        float(row["policy_pipeline_latency_ms"])
        for row in all_rows
        if int(row.get("tick", -1)) >= cold_start_ticks
        and isinstance(row.get("policy_pipeline_latency_ms"), (int, float))
    ]
    maximum_cold_latency = max(cold_latencies, default=float("inf"))
    maximum_steady_latency = max(steady_latencies, default=float("inf"))
    collisions = sum(int(episode.get("collision_events", 0)) for episode in episodes)
    lane_invasions = sum(
        int(episode.get("lane_invasion_events", 0)) for episode in episodes
    )
    red_light_violations = sum(
        int(episode.get("red_light_violations", 0)) for episode in episodes
    )
    no_progress = sum(
        episode.get("terminated_reason") == "no_progress" for episode in episodes
    )
    maximum_stationary = max(
        (float(episode.get("stationary_fraction", 1.0)) for episode in episodes),
        default=1.0,
    )
    maximum_speed = max(
        (float(episode.get("max_speed_mps", float("inf"))) for episode in episodes),
        default=float("inf"),
    )
    maximum_offset = max(
        (float(episode.get("max_abs_lane_offset_m", float("inf"))) for episode in episodes),
        default=float("inf"),
    )
    recovery_active_at_end = any(
        bool(telemetry_by_seed[seed][-1].get("recovery_active"))
        for seed in expected_seeds
        if telemetry_by_seed.get(seed)
    )
    liveness_active_at_end = any(
        bool(telemetry_by_seed[seed][-1].get("liveness_active"))
        for seed in expected_seeds
        if telemetry_by_seed.get(seed)
    )
    checks = {
        "report_status": report.get("status") == "passed",
        "model_contract": report.get("model_type")
        == "factorized_temporal_policy_phase6_safety_arbitration",
        "checkpoint_identity": report.get("checkpoint_sha256")
        == expected_checkpoint_sha256,
        "expected_seeds": observed_seeds == expected_seeds,
        "telemetry_complete": telemetry_complete,
        "tick_completion": len(episodes) == len(expected_seeds)
        and all(int(episode.get("ticks_completed", 0)) >= minimum_ticks for episode in episodes),
        "collision_budget": collisions <= maximum_collisions,
        "lane_invasion_budget": lane_invasions <= maximum_lane_invasions,
        "red_light_budget": red_light_violations <= maximum_red_light_violations,
        "liveness_termination_budget": no_progress <= maximum_no_progress_terminations,
        "route_progress": float(aggregate.get("mean_route_progress_fraction", 0.0))
        >= minimum_mean_route_progress_fraction,
        "distance": float(aggregate.get("mean_distance_traveled_m", 0.0))
        >= minimum_mean_distance_m,
        "stationary_fraction": maximum_stationary <= maximum_stationary_fraction,
        "speed_envelope": maximum_speed <= maximum_speed_mps,
        "lane_offset_envelope": maximum_offset <= maximum_lane_offset_m,
        "recovery_disengaged": telemetry_complete and not recovery_active_at_end,
        "liveness_disengaged": telemetry_complete and not liveness_active_at_end,
        "cold_start_latency": maximum_cold_latency
        <= maximum_cold_start_latency_ms,
        "steady_state_latency": maximum_steady_latency
        <= maximum_steady_state_latency_ms,
    }
    failures = [name for name, passed in checks.items() if not passed]
    replan_ticks = sum(bool(row.get("factorized_policy_replanned")) for row in all_rows)
    safety_ticks = sum(bool(row.get("safety_active")) for row in all_rows)
    steering_override_ticks = sum(
        bool(row.get("final_steering_direction_overridden")) for row in all_rows
    )
    liveness_ticks = sum(bool(row.get("liveness_active")) for row in all_rows)
    total_ticks = len(all_rows)
    return {
        "status": "passed" if not failures else "failed",
        "gate_passed": not failures,
        "checks": checks,
        "failures": failures,
        "metrics": {
            "episodes": len(episodes),
            "seeds": observed_seeds,
            "mean_route_progress_fraction": float(
                aggregate.get("mean_route_progress_fraction", 0.0)
            ),
            "mean_distance_traveled_m": float(
                aggregate.get("mean_distance_traveled_m", 0.0)
            ),
            "total_collisions": collisions,
            "total_lane_invasions": lane_invasions,
            "total_red_light_violations": red_light_violations,
            "no_progress_terminations": no_progress,
            "maximum_stationary_fraction": maximum_stationary,
            "maximum_speed_mps": maximum_speed,
            "maximum_abs_lane_offset_m": maximum_offset,
            "maximum_cold_start_policy_latency_ms": maximum_cold_latency,
            "maximum_steady_state_policy_latency_ms": maximum_steady_latency,
            "factorized_replan_ticks": replan_ticks,
            "safety_active_ticks": safety_ticks,
            "safety_active_fraction": safety_ticks / total_ticks if total_ticks else 0.0,
            "steering_override_ticks": steering_override_ticks,
            "liveness_active_ticks": liveness_ticks,
            "recovery_active_at_end": recovery_active_at_end,
            "liveness_active_at_end": liveness_active_at_end,
        },
        "claim_boundary": (
            "This gate evaluates the factorized policy with frozen deterministic "
            "Phase 6 safety and liveness arbitration; telemetry discloses intervention."
        ),
    }


def main() -> int:
    args = parse_args()
    config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    contract = config["phase7_closed_loop_evaluation"]
    report = json.loads(args.report.read_text(encoding="utf-8"))
    telemetry_by_seed: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(args.telemetry_dir.glob("*.jsonl")):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if rows:
            telemetry_by_seed[int(rows[0]["seed"])] = rows
    seed = int(contract["seed"])
    expected_seeds = [seed + index for index in range(int(contract["episodes"]))]
    result = evaluate_acceptance(
        report,
        telemetry_by_seed,
        expected_seeds=expected_seeds,
        expected_checkpoint_sha256=str(contract["checkpoint_sha256"]),
        minimum_ticks=int(contract["ticks_per_episode"]),
        minimum_mean_route_progress_fraction=float(
            contract["minimum_mean_route_progress_fraction"]
        ),
        minimum_mean_distance_m=float(contract["minimum_mean_distance_m"]),
        maximum_collisions=int(contract["maximum_collisions"]),
        maximum_lane_invasions=int(contract["maximum_lane_invasions"]),
        maximum_red_light_violations=int(contract["maximum_red_light_violations"]),
        maximum_no_progress_terminations=int(
            contract["maximum_no_progress_terminations"]
        ),
        maximum_stationary_fraction=float(contract["maximum_stationary_fraction"]),
        maximum_speed_mps=float(contract["maximum_speed_mps"]),
        maximum_lane_offset_m=float(contract["maximum_lane_offset_m"]),
        cold_start_ticks=int(contract["cold_start_ticks"]),
        maximum_cold_start_latency_ms=float(
            contract["maximum_cold_start_latency_ms"]
        ),
        maximum_steady_state_latency_ms=float(
            contract["maximum_steady_state_latency_ms"]
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
