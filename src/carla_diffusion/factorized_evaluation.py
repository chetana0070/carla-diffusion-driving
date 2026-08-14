"""Promotion contract for the factorized temporal policy."""

from __future__ import annotations

from typing import Any

from .diffusion_evaluation import promotion_decision


def factorized_promotion_decision(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    predicted_smoothness: dict[str, float],
    target_smoothness: dict[str, float],
    expected_validation_samples: int,
    validation_samples: int,
    expected_test_samples: int,
    maximum_relative_rmse: float,
    maximum_absolute_longitudinal_bias: float,
    maximum_longitudinal_behavior_drift: float,
    maximum_longitudinal_smoothness_ratio: float,
    latency_gate_passed: bool,
) -> dict[str, Any]:
    decision = promotion_decision(
        candidate,
        baseline,
        expected_validation_samples=expected_validation_samples,
        validation_samples=validation_samples,
        expected_test_samples=expected_test_samples,
        maximum_relative_rmse=maximum_relative_rmse,
        maximum_absolute_longitudinal_bias=maximum_absolute_longitudinal_bias,
        maximum_longitudinal_behavior_drift=maximum_longitudinal_behavior_drift,
        latency_gate_passed=latency_gate_passed,
    )
    predicted_step = predicted_smoothness["mean_abs_longitudinal_step"]
    target_step = target_smoothness["mean_abs_longitudinal_step"]
    ratio = predicted_step / max(target_step, 1e-12)
    smoothness_passed = ratio <= maximum_longitudinal_smoothness_ratio
    decision["checks"]["longitudinal_chunk_smoothness"] = smoothness_passed
    if not smoothness_passed:
        decision["failures"].append("longitudinal_chunk_smoothness")
    decision["gate_passed"] = not decision["failures"]
    decision["longitudinal_smoothness_ratio"] = ratio
    return decision
