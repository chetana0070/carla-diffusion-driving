"""Compact language-conditioned action planner for the Phase 8 preflight."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class HierarchicalVLAPlanner(nn.Module):
    """Fuse temporal vision, vehicle state, and instruction tokens into an action chunk."""

    def __init__(
        self,
        *,
        vocabulary_size: int,
        state_dimension: int,
        action_horizon: int,
        language_dimension: int = 128,
        state_hidden_dimension: int = 128,
        fusion_dimension: int = 256,
        pretrained_visual_encoder: bool = False,
    ) -> None:
        super().__init__()
        if vocabulary_size < 4 or state_dimension < 1 or action_horizon < 1:
            raise ValueError("invalid VLA model dimensions")
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained_visual_encoder else None
        backbone = efficientnet_b0(weights=weights)
        self.visual_encoder = backbone.features
        self.visual_pool = nn.AdaptiveAvgPool2d(1)
        self.visual_projection = nn.Linear(1280, fusion_dimension)
        self.token_embedding = nn.Embedding(vocabulary_size, language_dimension, padding_idx=0)
        self.language_projection = nn.Linear(language_dimension, fusion_dimension)
        self.state_encoder = nn.GRU(
            input_size=state_dimension,
            hidden_size=state_hidden_dimension,
            batch_first=True,
        )
        self.state_projection = nn.Linear(state_hidden_dimension, fusion_dimension)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dimension * 3),
            nn.Linear(fusion_dimension * 3, fusion_dimension),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dimension, action_horizon * 2),
        )
        self.action_horizon = action_horizon

    def forward(
        self,
        images: torch.Tensor,
        state_history: torch.Tensor,
        instruction_token_ids: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError("images must have shape BxTxCxHxW")
        batch, history, channels, height, width = images.shape
        encoded_images = self.visual_encoder(
            images.reshape(batch * history, channels, height, width)
        )
        visual = self.visual_pool(encoded_images).flatten(1).reshape(batch, history, -1)
        visual = self.visual_projection(visual.mean(dim=1))

        embedded_tokens = self.token_embedding(instruction_token_ids)
        mask = instruction_attention_mask.unsqueeze(-1)
        language = (embedded_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        language = self.language_projection(language)

        _, state_hidden = self.state_encoder(state_history)
        state = self.state_projection(state_hidden[-1])
        actions = self.fusion(torch.cat((visual, state, language), dim=-1))
        return torch.tanh(actions.reshape(batch, self.action_horizon, 2))


def weighted_action_chunk_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share BxHx2 shape")
    per_sample = torch.mean((prediction - target) ** 2, dim=(1, 2))
    return torch.sum(per_sample * weight) / weight.sum().clamp_min(1e-8)


def corrective_action_chunk_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    condition_weight: torch.Tensor,
    *,
    chunk_steering_weight: float,
    chunk_longitudinal_weight: float,
    first_steering_weight: float,
    first_longitudinal_weight: float,
    steering_derivative_weight: float,
    longitudinal_derivative_weight: float,
    longitudinal_bias_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize the Phase 8.2 promotion failures without using test data."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share BxHx2 shape")
    if prediction.shape[1] < 2 or prediction.shape[2] != 2:
        raise ValueError("corrective loss requires at least two BxHx2 actions")
    if (
        sample_weight.shape != prediction.shape[:1]
        or condition_weight.shape != prediction.shape[:1]
    ):
        raise ValueError("weights must have one value per sample")
    objective_weights = (
        chunk_steering_weight,
        chunk_longitudinal_weight,
        first_steering_weight,
        first_longitudinal_weight,
        steering_derivative_weight,
        longitudinal_derivative_weight,
        longitudinal_bias_weight,
    )
    if any(value < 0 for value in objective_weights):
        raise ValueError("corrective objective weights cannot be negative")
    combined_weight = sample_weight * condition_weight
    if not bool(torch.all(torch.isfinite(combined_weight))) or bool(
        torch.any(combined_weight <= 0)
    ):
        raise ValueError("combined sample weights must be finite and positive")
    normalized_weight = combined_weight / combined_weight.sum().clamp_min(1e-8)

    error = prediction - target
    prediction_derivative = prediction[:, 1:] - prediction[:, :-1]
    target_derivative = target[:, 1:] - target[:, :-1]
    derivative_error = prediction_derivative - target_derivative

    def weighted_mean(per_sample: torch.Tensor) -> torch.Tensor:
        return torch.sum(per_sample * normalized_weight)

    components = {
        "chunk_steering_mse": weighted_mean(torch.mean(error[:, :, 0] ** 2, dim=1)),
        "chunk_longitudinal_mse": weighted_mean(
            torch.mean(error[:, :, 1] ** 2, dim=1)
        ),
        "first_action_steering_mse": weighted_mean(error[:, 0, 0] ** 2),
        "first_action_longitudinal_mse": weighted_mean(error[:, 0, 1] ** 2),
        "steering_derivative_mse": weighted_mean(
            torch.mean(derivative_error[:, :, 0] ** 2, dim=1)
        ),
        "longitudinal_derivative_mse": weighted_mean(
            torch.mean(derivative_error[:, :, 1] ** 2, dim=1)
        ),
        "longitudinal_bias_squared": torch.sum(
            error[:, 0, 1] * normalized_weight
        )
        ** 2,
    }
    loss = (
        chunk_steering_weight * components["chunk_steering_mse"]
        + chunk_longitudinal_weight * components["chunk_longitudinal_mse"]
        + first_steering_weight * components["first_action_steering_mse"]
        + first_longitudinal_weight * components["first_action_longitudinal_mse"]
        + steering_derivative_weight * components["steering_derivative_mse"]
        + longitudinal_derivative_weight
        * components["longitudinal_derivative_mse"]
        + longitudinal_bias_weight * components["longitudinal_bias_squared"]
    )
    return loss, components


def corrective_selection_score(
    *,
    joint_rmse: float,
    longitudinal_bias: float,
    steering_smoothness_ratio: float,
    target_maximum_absolute_bias: float,
    target_maximum_smoothness_ratio: float,
    bias_violation_weight: float,
    smoothness_violation_weight: float,
) -> tuple[float, dict[str, float]]:
    """Rank checkpoints on validation accuracy and promotion-headroom violations."""
    if joint_rmse < 0 or target_maximum_absolute_bias <= 0:
        raise ValueError("invalid corrective selection metric")
    if target_maximum_smoothness_ratio < 1:
        raise ValueError("smoothness target must be at least one")
    bias_violation = max(
        0.0, abs(longitudinal_bias) / target_maximum_absolute_bias - 1.0
    )
    smoothness_violation = max(
        0.0, steering_smoothness_ratio / target_maximum_smoothness_ratio - 1.0
    )
    score = (
        joint_rmse
        + bias_violation_weight * bias_violation
        + smoothness_violation_weight * smoothness_violation
    )
    return score, {
        "joint_rmse": joint_rmse,
        "absolute_longitudinal_bias": abs(longitudinal_bias),
        "steering_smoothness_ratio": steering_smoothness_ratio,
        "bias_violation": bias_violation,
        "smoothness_violation": smoothness_violation,
    }
