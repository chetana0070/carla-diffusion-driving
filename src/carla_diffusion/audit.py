"""Semantic coverage metrics for expert-driving datasets."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any


def quantile(values: Sequence[float], probability: float) -> float | None:
    """Return a linearly interpolated quantile without NumPy."""
    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "p05": quantile(values, 0.05),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def fraction(count: int, total: int) -> float:
    return count / total if total else 0.0


def counter_report(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    return {
        key: {"count": value, "fraction": fraction(value, total)}
        for key, value in sorted(counter.items())
    }


def usable_temporal_windows(samples_per_episode: Iterable[int], history: int, horizon: int) -> int:
    """Count windows with `history` observations and `horizon` future actions."""
    if history < 1 or horizon < 1:
        raise ValueError("history and horizon must be positive")
    return sum(max(0, samples - history - horizon + 2) for samples in samples_per_episode)


def evaluate_readiness(metrics: dict[str, Any]) -> list[dict[str, object]]:
    """Apply pilot gates that determine whether targeted recollection is needed."""
    commands = metrics["categorical"]["route_command"]
    gates = [
        (
            "minimum_samples",
            metrics["samples"] >= 6000,
            metrics["samples"],
            ">= 6000",
            True,
        ),
        (
            "minimum_episodes",
            metrics["episodes"] >= 10,
            metrics["episodes"],
            ">= 10",
            True,
        ),
        (
            "temporal_windows",
            metrics["usable_temporal_windows"] >= 5000,
            metrics["usable_temporal_windows"],
            ">= 5000",
            True,
        ),
        (
            "image_integrity",
            metrics["image_integrity"]["failure_count"] == 0
            and metrics["image_integrity"]["dimensions"] == {"640x360": metrics["samples"]}
            and metrics["image_integrity"]["formats"] == {"JPEG": metrics["samples"]},
            {
                "failures": metrics["image_integrity"]["failure_count"],
                "dimensions": metrics["image_integrity"]["dimensions"],
                "formats": metrics["image_integrity"]["formats"],
            },
            "0 corrupt images; all images JPEG 640x360",
            True,
        ),
        (
            "steering_diversity",
            metrics["actions"]["absolute_steering"]["p95"] >= 0.08,
            metrics["actions"]["absolute_steering"]["p95"],
            ">= 0.08 p95 absolute steering",
            True,
        ),
        (
            "active_braking_coverage",
            metrics["actions"]["active_braking_fraction"] >= 0.01,
            metrics["actions"]["active_braking_fraction"],
            ">= 1% while ego speed is at least 0.5 m/s",
            True,
        ),
        (
            "moving_coverage",
            metrics["states"]["moving_fraction"] >= 0.50,
            metrics["states"]["moving_fraction"],
            ">= 50%",
            True,
        ),
        (
            "route_command_coverage",
            all(
                commands.get(name, {}).get("count", 0) >= 10
                for name in ("left", "right", "straight")
            ),
            {
                name: commands.get(name, {}).get("count", 0)
                for name in ("left", "right", "straight")
            },
            ">= 10 samples for left, right, and straight",
            True,
        ),
        (
            "lead_vehicle_context",
            metrics["states"]["lead_vehicle_available_fraction"] >= 0.01,
            metrics["states"]["lead_vehicle_available_fraction"],
            ">= 1%",
            False,
        ),
        (
            "lead_vehicle_distance_integrity",
            metrics["states"]["lead_vehicle_distance_outlier_count"] == 0,
            metrics["states"]["lead_vehicle_distance_outlier_count"],
            "0 lead-vehicle distances beyond 80 m",
            True,
        ),
        (
            "stationary_brake_hold_balance",
            metrics["actions"]["stationary_brake_hold_fraction"] <= 0.40,
            metrics["actions"]["stationary_brake_hold_fraction"],
            "<= 40% (otherwise down-weight redundant hold windows during training)",
            False,
        ),
        (
            "acceleration_outlier_quality",
            metrics["states"]["acceleration_outlier_fraction"] <= 0.01,
            metrics["states"]["acceleration_outlier_fraction"],
            "<= 1% with |longitudinal acceleration| > 12 m/s^2",
            False,
        ),
        (
            "traffic_light_context",
            metrics["states"]["traffic_light_available_fraction"] >= 0.005,
            metrics["states"]["traffic_light_available_fraction"],
            ">= 0.5%",
            False,
        ),
        (
            "collision_quality",
            metrics["events"]["collision"]["fraction"] <= 0.001,
            metrics["events"]["collision"]["fraction"],
            "<= 0.1%",
            True,
        ),
    ]
    return [
        {
            "name": name,
            "status": "passed" if passed else "failed",
            "observed": observed,
            "threshold": threshold,
            "required": required,
        }
        for name, passed, observed, threshold, required in gates
    ]
