"""Offline metrics and promotion contract for the hierarchical VLA planner."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .diffusion_evaluation import promotion_decision, summarize_action_pairs


def summarize_instruction_slices(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    route_commands: Sequence[str],
    traffic_light_states: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Summarize first-action quality for every observed language condition."""
    count = len(predictions)
    if not (
        count == len(targets) == len(route_commands) == len(traffic_light_states)
    ) or count == 0:
        raise ValueError("instruction slices require aligned, non-empty inputs")

    def grouped(labels: Sequence[str]) -> dict[str, Any]:
        members: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            members[str(label)].append(index)
        return {
            label: summarize_action_pairs(
                [predictions[index] for index in indices],
                [targets[index] for index in indices],
            )
            for label, indices in sorted(members.items())
        }

    return {
        "route_command": grouped(route_commands),
        "traffic_light_state": grouped(traffic_light_states),
    }


def vla_promotion_decision(
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
    maximum_chunk_smoothness_ratio: float,
    latency_gate_passed: bool,
    checkpoint_integrity_passed: bool,
    baseline_provenance_passed: bool,
    instruction_slice_coverage_passed: bool,
) -> dict[str, Any]:
    """Apply the Phase 8.1 promotion contract without expanding its claim scope."""
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
    ratios = {
        "steering": predicted_smoothness["mean_abs_steering_step"]
        / max(target_smoothness["mean_abs_steering_step"], 1e-12),
        "longitudinal": predicted_smoothness["mean_abs_longitudinal_step"]
        / max(target_smoothness["mean_abs_longitudinal_step"], 1e-12),
    }
    additional_checks = {
        "chunk_smoothness": max(ratios.values()) <= maximum_chunk_smoothness_ratio,
        "checkpoint_integrity": checkpoint_integrity_passed,
        "baseline_provenance": baseline_provenance_passed,
        "instruction_slice_coverage": instruction_slice_coverage_passed,
    }
    decision["checks"].update(additional_checks)
    decision["failures"] = [
        name for name, passed in decision["checks"].items() if not passed
    ]
    decision["gate_passed"] = not decision["failures"]
    decision["chunk_smoothness_ratio"] = ratios
    return decision
