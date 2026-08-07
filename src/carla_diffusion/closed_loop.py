"""Dependency-light contracts for closed-loop policy evaluation."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any


class NoProgressMonitor:
    """Detect sustained lack of traveled distance without reacting to brief stops."""

    def __init__(self, window_ticks: int, minimum_distance_m: float) -> None:
        if window_ticks < 2 or not math.isfinite(minimum_distance_m) or minimum_distance_m <= 0:
            raise ValueError("invalid no-progress monitor contract")
        self.window_ticks = window_ticks
        self.minimum_distance_m = minimum_distance_m
        self._distances: deque[float] = deque(maxlen=window_ticks)

    def reset(self) -> None:
        self._distances.clear()

    def update(self, cumulative_distance_m: float) -> bool:
        if not math.isfinite(cumulative_distance_m) or cumulative_distance_m < 0:
            raise ValueError("cumulative distance must be finite and non-negative")
        self._distances.append(cumulative_distance_m)
        if len(self._distances) < self.window_ticks:
            return False
        return self._distances[-1] - self._distances[0] < self.minimum_distance_m


def route_command_from_geometry(
    current_yaw_degrees: float,
    future_yaw_degrees: float,
    *,
    junction_ahead: bool,
    turn_threshold_degrees: float = 15.0,
) -> str:
    if turn_threshold_degrees <= 0:
        raise ValueError("turn threshold must be positive")
    difference = (future_yaw_degrees - current_yaw_degrees + 180.0) % 360.0 - 180.0
    if not junction_ahead:
        return "follow_lane"
    # CARLA/Unreal uses a left-handed coordinate system: positive yaw turns right.
    if difference > turn_threshold_degrees:
        return "right"
    if difference < -turn_threshold_degrees:
        return "left"
    return "straight"


def longitudinal_to_pedals(longitudinal: float) -> tuple[float, float]:
    if not math.isfinite(longitudinal):
        raise ValueError("longitudinal action must be finite")
    bounded = max(-1.0, min(1.0, longitudinal))
    return (bounded, 0.0) if bounded >= 0.0 else (0.0, -bounded)


def latency_summary(milliseconds: Sequence[float]) -> dict[str, float]:
    if not milliseconds or not all(math.isfinite(value) and value >= 0 for value in milliseconds):
        raise ValueError("latencies must be a non-empty finite non-negative sequence")
    ordered = sorted(milliseconds)

    def percentile(fraction: float) -> float:
        index = round(fraction * (len(ordered) - 1))
        return ordered[index]

    return {
        "mean_ms": sum(ordered) / len(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1],
    }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_episode_reports(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("at least one episode report is required")
    return {
        "episodes": len(episodes),
        "mean_route_progress_fraction": mean(
            [float(episode["route_progress_fraction"]) for episode in episodes]
        ),
        "mean_distance_traveled_m": mean(
            [float(episode["distance_traveled_m"]) for episode in episodes]
        ),
        "total_collisions": sum(int(episode["collision_events"]) for episode in episodes),
        "total_lane_invasions": sum(
            int(episode["lane_invasion_events"]) for episode in episodes
        ),
        "total_red_light_violations": sum(
            int(episode["red_light_violations"]) for episode in episodes
        ),
        "mean_speed_mps": mean([float(episode["mean_speed_mps"]) for episode in episodes]),
        "mean_abs_steering_rate_per_second": mean(
            [float(episode["mean_abs_steering_rate_per_second"]) for episode in episodes]
        ),
        "mean_abs_longitudinal_rate_per_second": mean(
            [float(episode["mean_abs_longitudinal_rate_per_second"]) for episode in episodes]
        ),
    }


def evaluate_expert_oracle_gate(
    aggregate: Mapping[str, Any],
    termination_counts: Mapping[str, int],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the versioned Phase 5 harness-capability gate."""
    checks = {
        "route_progress_gate_passed": (
            float(aggregate["mean_route_progress_fraction"])
            >= float(contract["expert_oracle_min_route_progress_fraction"])
        ),
        "distance_gate_passed": (
            float(aggregate["mean_distance_traveled_m"])
            >= float(contract["expert_oracle_min_mean_distance_m"])
        ),
        "collision_gate_passed": (
            int(aggregate["total_collisions"])
            <= int(contract["expert_oracle_max_collisions"])
        ),
        "lane_invasion_gate_passed": (
            int(aggregate["total_lane_invasions"])
            <= int(contract["expert_oracle_max_lane_invasions"])
        ),
        "red_light_gate_passed": (
            int(aggregate["total_red_light_violations"])
            <= int(contract["expert_oracle_max_red_light_violations"])
        ),
        "no_progress_gate_passed": (
            int(termination_counts.get("no_progress", 0))
            <= int(contract["expert_oracle_max_no_progress_terminations"])
        ),
    }
    return {
        "protocol_version": str(contract["expert_oracle_protocol_version"]),
        "passed": all(checks.values()),
        "thresholds": {
            "minimum_mean_route_progress_fraction": contract[
                "expert_oracle_min_route_progress_fraction"
            ],
            "minimum_mean_distance_m": contract["expert_oracle_min_mean_distance_m"],
            "maximum_collisions": contract["expert_oracle_max_collisions"],
            "maximum_lane_invasions": contract["expert_oracle_max_lane_invasions"],
            "maximum_red_light_violations": contract[
                "expert_oracle_max_red_light_violations"
            ],
            "maximum_no_progress_terminations": contract[
                "expert_oracle_max_no_progress_terminations"
            ],
        },
        **checks,
    }
