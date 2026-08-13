"""Metric and promotion contracts for offline diffusion-policy evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Any


def summarize_action_pairs(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    *,
    neutral_threshold: float = 0.05,
) -> dict[str, Any]:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must be non-empty and equally sized")
    if neutral_threshold < 0:
        raise ValueError("neutral threshold must be non-negative")
    errors: list[list[float]] = [[], []]
    predicted: list[list[float]] = [[], []]
    expected: list[list[float]] = [[], []]
    out_of_bounds = 0
    for prediction, target in zip(predictions, targets, strict=True):
        if len(prediction) != 2 or len(target) != 2:
            raise ValueError("every action must contain steering and longitudinal values")
        for dimension in range(2):
            value = float(prediction[dimension])
            truth = float(target[dimension])
            if not math.isfinite(value) or not math.isfinite(truth):
                raise ValueError("action metrics require finite values")
            predicted[dimension].append(value)
            expected[dimension].append(truth)
            errors[dimension].append(value - truth)
            out_of_bounds += int(abs(value) > 1.0 + 1e-6)

    def dimension_summary(index: int) -> dict[str, float]:
        count = len(errors[index])
        return {
            "mae": sum(abs(value) for value in errors[index]) / count,
            "rmse": math.sqrt(sum(value * value for value in errors[index]) / count),
            "bias": sum(errors[index]) / count,
            "prediction_mean": sum(predicted[index]) / count,
            "target_mean": sum(expected[index]) / count,
            "prediction_min": min(predicted[index]),
            "prediction_max": max(predicted[index]),
        }

    longitudinal_predictions = predicted[1]
    longitudinal_targets = expected[1]

    def behavior(values: Sequence[float]) -> dict[str, float]:
        count = len(values)
        return {
            "acceleration_fraction": sum(
                value > neutral_threshold for value in values
            )
            / count,
            "braking_fraction": sum(value < -neutral_threshold for value in values)
            / count,
            "neutral_fraction": sum(
                abs(value) <= neutral_threshold for value in values
            )
            / count,
        }

    steering = dimension_summary(0)
    longitudinal = dimension_summary(1)
    return {
        "samples": len(predictions),
        "steering": steering,
        "longitudinal": longitudinal,
        "joint_rmse": math.sqrt(
            (steering["rmse"] ** 2 + longitudinal["rmse"] ** 2) / 2
        ),
        "out_of_bounds_fraction": out_of_bounds / (len(predictions) * 2),
        "predicted_longitudinal_behavior": behavior(longitudinal_predictions),
        "target_longitudinal_behavior": behavior(longitudinal_targets),
    }


def summarize_chunk_smoothness(
    chunks: Sequence[Sequence[Sequence[float]]],
) -> dict[str, float]:
    if not chunks:
        raise ValueError("at least one action chunk is required")
    differences: list[list[float]] = [[], []]
    for chunk in chunks:
        if len(chunk) < 2:
            raise ValueError("action chunks must contain at least two actions")
        for previous, current in pairwise(chunk):
            if len(previous) != 2 or len(current) != 2:
                raise ValueError("every chunk action must contain two values")
            for dimension in range(2):
                differences[dimension].append(
                    abs(float(current[dimension]) - float(previous[dimension]))
                )
    return {
        "mean_abs_steering_step": sum(differences[0]) / len(differences[0]),
        "mean_abs_longitudinal_step": sum(differences[1]) / len(differences[1]),
        "maximum_abs_steering_step": max(differences[0]),
        "maximum_abs_longitudinal_step": max(differences[1]),
    }


def promotion_decision(
    diffusion: dict[str, Any],
    baseline: dict[str, Any],
    *,
    expected_validation_samples: int,
    validation_samples: int,
    expected_test_samples: int,
    maximum_relative_rmse: float,
    maximum_absolute_longitudinal_bias: float,
    maximum_longitudinal_behavior_drift: float,
    latency_gate_passed: bool,
) -> dict[str, Any]:
    behavior_drifts = {
        key: abs(
            diffusion["predicted_longitudinal_behavior"][key]
            - diffusion["target_longitudinal_behavior"][key]
        )
        for key in ("acceleration_fraction", "braking_fraction", "neutral_fraction")
    }
    relative = {
        "steering": diffusion["steering"]["rmse"]
        / max(baseline["steering"]["rmse"], 1e-12),
        "longitudinal": diffusion["longitudinal"]["rmse"]
        / max(baseline["longitudinal"]["rmse"], 1e-12),
        "joint": diffusion["joint_rmse"] / max(baseline["joint_rmse"], 1e-12),
    }
    checks = {
        "validation_coverage": validation_samples == expected_validation_samples,
        "test_coverage": diffusion["samples"] == expected_test_samples,
        "bounded_actions": diffusion["out_of_bounds_fraction"] == 0,
        "steering_rmse": relative["steering"] <= maximum_relative_rmse,
        "longitudinal_rmse": relative["longitudinal"] <= maximum_relative_rmse,
        "joint_rmse": relative["joint"] <= maximum_relative_rmse,
        "longitudinal_bias": abs(diffusion["longitudinal"]["bias"])
        <= maximum_absolute_longitudinal_bias,
        "longitudinal_behavior": max(behavior_drifts.values())
        <= maximum_longitudinal_behavior_drift,
        "latency": latency_gate_passed,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "gate_passed": not failures,
        "checks": checks,
        "failures": failures,
        "relative_rmse_to_temporal_bc": relative,
        "longitudinal_behavior_drift": behavior_drifts,
    }
