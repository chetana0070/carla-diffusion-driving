"""Validation-calibrated steering smoothing for hierarchical VLA chunks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch

from .diffusion_evaluation import summarize_action_pairs, summarize_chunk_smoothness

SMOOTHER_TYPE = "causal_exponential_steering_smoother"


def causal_steering_smoother(actions: torch.Tensor, alpha: float) -> torch.Tensor:
    """Smooth future steering causally while preserving action zero and longitudinal."""
    if actions.ndim < 2 or actions.shape[-1] != 2 or actions.shape[-2] < 1:
        raise ValueError("actions must end in an Hx2 action chunk")
    if not 0 < alpha <= 1:
        raise ValueError("steering smoothing alpha must be in (0, 1]")
    if not bool(torch.all(torch.isfinite(actions))):
        raise ValueError("action chunks must be finite")
    smoothed = actions.clone()
    for index in range(1, actions.shape[-2]):
        previous = smoothed[..., index - 1, 0]
        current = actions[..., index, 0]
        smoothed[..., index, 0] = previous + alpha * (current - previous)
    return smoothed


def checkpoint_smoother_alpha(checkpoint: Mapping[str, Any]) -> float | None:
    """Return the frozen alpha, or None for an untransformed base checkpoint."""
    transform = checkpoint.get("deployment_transform")
    if transform is None:
        return None
    if not isinstance(transform, Mapping):
        raise TypeError("checkpoint deployment transform must be a mapping")
    if transform.get("type") != SMOOTHER_TYPE:
        raise ValueError("checkpoint uses an unsupported deployment transform")
    alpha = float(transform["alpha"])
    if not 0 < alpha <= 1:
        raise ValueError("checkpoint steering smoothing alpha must be in (0, 1]")
    if transform.get("preserve_first_action") is not True:
        raise ValueError("checkpoint smoother must preserve the first action")
    return alpha


def apply_checkpoint_smoother(
    actions: torch.Tensor,
    checkpoint: Mapping[str, Any],
) -> torch.Tensor:
    """Apply the checkpoint-bound postprocessor when one is declared."""
    alpha = checkpoint_smoother_alpha(checkpoint)
    return actions if alpha is None else causal_steering_smoother(actions, alpha)


def smooth_action_chunks(
    chunks: Sequence[Sequence[Sequence[float]]],
    alpha: float,
) -> list[list[list[float]]]:
    """List-oriented smoothing helper used by deterministic calibration."""
    tensor = torch.tensor(chunks, dtype=torch.float64)
    return cast(
        list[list[list[float]]],
        causal_steering_smoother(tensor, alpha).tolist(),
    )


def select_steering_smoothing(
    predicted_chunks: Sequence[Sequence[Sequence[float]]],
    target_chunks: Sequence[Sequence[Sequence[float]]],
    candidate_alphas: Sequence[float],
    *,
    target_maximum_smoothness_ratio: float,
    maximum_full_chunk_steering_rmse_degradation_fraction: float,
) -> dict[str, Any]:
    """Choose the least invasive alpha using validation chunks only."""
    if len(predicted_chunks) != len(target_chunks) or not predicted_chunks:
        raise ValueError("calibration chunks must be aligned and non-empty")
    if target_maximum_smoothness_ratio < 1:
        raise ValueError("smoothness target must be at least one")
    if maximum_full_chunk_steering_rmse_degradation_fraction < 0:
        raise ValueError("RMSE degradation allowance must be non-negative")
    alphas = [float(value) for value in candidate_alphas]
    if not alphas or any(not 0 < value <= 1 for value in alphas):
        raise ValueError("candidate alphas must be non-empty and in (0, 1]")
    if len(set(alphas)) != len(alphas):
        raise ValueError("candidate alphas must be unique")

    flat_targets = [action for chunk in target_chunks for action in chunk]
    flat_raw = [action for chunk in predicted_chunks for action in chunk]
    raw_metrics = summarize_action_pairs(flat_raw, flat_targets)
    target_smoothness = summarize_chunk_smoothness(target_chunks)
    target_step = max(target_smoothness["mean_abs_steering_step"], 1e-12)
    raw_steering_rmse = max(float(raw_metrics["steering"]["rmse"]), 1e-12)
    candidates: list[dict[str, Any]] = []

    for alpha in sorted(alphas, reverse=True):
        smoothed = smooth_action_chunks(predicted_chunks, alpha)
        flat_smoothed = [action for chunk in smoothed for action in chunk]
        metrics = summarize_action_pairs(flat_smoothed, flat_targets)
        smoothness = summarize_chunk_smoothness(smoothed)
        ratio = smoothness["mean_abs_steering_step"] / target_step
        degradation = float(metrics["steering"]["rmse"]) / raw_steering_rmse - 1.0
        first_action_maximum_delta = max(
            abs(float(before[0][dimension]) - float(after[0][dimension]))
            for before, after in zip(predicted_chunks, smoothed, strict=True)
            for dimension in range(2)
        )
        longitudinal_maximum_delta = max(
            abs(float(before[index][1]) - float(after[index][1]))
            for before, after in zip(predicted_chunks, smoothed, strict=True)
            for index in range(len(before))
        )
        eligible = (
            ratio <= target_maximum_smoothness_ratio
            and degradation
            <= maximum_full_chunk_steering_rmse_degradation_fraction
            and math.isclose(first_action_maximum_delta, 0.0, abs_tol=1e-12)
            and math.isclose(longitudinal_maximum_delta, 0.0, abs_tol=1e-12)
        )
        candidates.append(
            {
                "alpha": alpha,
                "eligible": eligible,
                "steering_smoothness_ratio": ratio,
                "full_chunk_steering_rmse": metrics["steering"]["rmse"],
                "full_chunk_steering_rmse_degradation_fraction": degradation,
                "first_action_maximum_delta": first_action_maximum_delta,
                "longitudinal_maximum_delta": longitudinal_maximum_delta,
            }
        )

    selected = next((candidate for candidate in candidates if candidate["eligible"]), None)
    return {
        "status": "passed" if selected is not None else "failed",
        "selection_rule": "largest eligible alpha",
        "selected_alpha": None if selected is None else selected["alpha"],
        "raw_full_chunk_steering_rmse": raw_metrics["steering"]["rmse"],
        "raw_steering_smoothness_ratio": (
            summarize_chunk_smoothness(predicted_chunks)["mean_abs_steering_step"]
            / target_step
        ),
        "target_mean_abs_steering_step": target_step,
        "target_maximum_smoothness_ratio": target_maximum_smoothness_ratio,
        "maximum_full_chunk_steering_rmse_degradation_fraction": (
            maximum_full_chunk_steering_rmse_degradation_fraction
        ),
        "candidates": candidates,
    }
