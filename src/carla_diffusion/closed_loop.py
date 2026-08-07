"""Dependency-light contracts for closed-loop policy evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


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
