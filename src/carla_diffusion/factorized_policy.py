"""Factorized temporal policy: diffusion steering and deterministic longitudinal control."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from .diffusion_policy import DiffusionSchedule, TemporalDiffusionPolicy


class FactorizedTemporalPolicy(nn.Module):
    """Share perception while assigning each control axis to its strongest model."""

    def __init__(
        self,
        *,
        history_frames: int = 4,
        action_horizon: int = 16,
        action_dimension: int = 2,
        state_dimension: int = 10,
        condition_dimension: int = 9,
        image_projection_dimension: int = 192,
        state_projection_dimension: int = 64,
        temporal_hidden_dimension: int = 256,
        condition_projection_dimension: int = 64,
        denoiser_dimension: int = 256,
        denoiser_layers: int = 3,
        denoiser_heads: int = 4,
        longitudinal_hidden_dimension: int = 256,
        dropout: float = 0.1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if action_dimension != 2:
            raise ValueError("factorized policy requires steering + longitudinal actions")
        if longitudinal_hidden_dimension <= 0:
            raise ValueError("longitudinal hidden dimension must be positive")
        self.history_frames = history_frames
        self.action_horizon = action_horizon
        self.action_dimension = action_dimension
        self.diffusion = TemporalDiffusionPolicy(
            history_frames=history_frames,
            action_horizon=action_horizon,
            action_dimension=action_dimension,
            state_dimension=state_dimension,
            condition_dimension=condition_dimension,
            image_projection_dimension=image_projection_dimension,
            state_projection_dimension=state_projection_dimension,
            temporal_hidden_dimension=temporal_hidden_dimension,
            condition_projection_dimension=condition_projection_dimension,
            denoiser_dimension=denoiser_dimension,
            denoiser_layers=denoiser_layers,
            denoiser_heads=denoiser_heads,
            dropout=dropout,
            pretrained=pretrained,
        )
        context_dimension = temporal_hidden_dimension + condition_projection_dimension
        self.longitudinal_trunk = nn.Sequential(
            nn.Linear(context_dimension, longitudinal_hidden_dimension),
            nn.LayerNorm(longitudinal_hidden_dimension),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(longitudinal_hidden_dimension, longitudinal_hidden_dimension),
            nn.SiLU(),
        )
        self.longitudinal_action_head = nn.Sequential(
            nn.Linear(longitudinal_hidden_dimension, action_horizon),
            nn.Tanh(),
        )
        self.longitudinal_mode_head = nn.Linear(
            longitudinal_hidden_dimension, action_horizon * 3
        )

    def set_encoder_trainable(self, trainable: bool) -> None:
        self.diffusion.set_encoder_trainable(trainable)

    def train(self, mode: bool = True) -> FactorizedTemporalPolicy:
        super().train(mode)
        if mode and not self.diffusion._encoder_trainable:
            self.diffusion.encoder.eval()
        return self

    def encode_observation(
        self,
        images: torch.Tensor,
        state_history: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.diffusion.encode_observation(images, state_history, condition)

    def denoise(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.diffusion.denoise(noisy_actions, timesteps, context)

    def predict_longitudinal(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.longitudinal_trunk(context)
        actions = self.longitudinal_action_head(features)
        logits = self.longitudinal_mode_head(features).reshape(
            context.shape[0], self.action_horizon, 3
        )
        return actions, logits

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        state_history: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.encode_observation(images, state_history, condition)
        noise = self.denoise(noisy_actions, timesteps, context)
        longitudinal, mode_logits = self.predict_longitudinal(context)
        return noise, longitudinal, mode_logits

    @torch.inference_mode()
    def sample(
        self,
        schedule: DiffusionSchedule,
        images: torch.Tensor,
        state_history: torch.Tensor,
        condition: torch.Tensor,
        *,
        inference_steps: int,
        eta: float = 0.0,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = self.encode_observation(images, state_history, condition)
        diffusion_chunk = schedule.ddim_sample(
            self.diffusion,
            images,
            state_history,
            condition,
            inference_steps=inference_steps,
            eta=eta,
            initial_noise=initial_noise,
            context=context,
        )
        longitudinal, _ = self.predict_longitudinal(context)
        return torch.stack((diffusion_chunk[:, :, 0], longitudinal), dim=2)


def load_diffusion_warm_start(
    model: FactorizedTemporalPolicy, checkpoint: dict[str, Any]
) -> None:
    if checkpoint.get("model_type") != "temporal_diffusion_policy":
        raise ValueError("warm-start checkpoint is not a temporal diffusion policy")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise TypeError("warm-start checkpoint has no model state")
    model.diffusion.load_state_dict(cast(dict[str, torch.Tensor], state), strict=True)


def longitudinal_modes(
    target: torch.Tensor, *, neutral_threshold: float = 0.05
) -> torch.Tensor:
    if target.ndim != 2 or neutral_threshold < 0:
        raise ValueError("longitudinal targets must be BxH with a valid threshold")
    modes = torch.ones_like(target, dtype=torch.long)
    modes[target < -neutral_threshold] = 0
    modes[target > neutral_threshold] = 2
    return modes


def weighted_longitudinal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    mode_weights: torch.Tensor,
    neutral_threshold: float = 0.05,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("longitudinal actions must be matching BxH tensors")
    if sample_weights.numel() != prediction.shape[0] or mode_weights.shape != (3,):
        raise ValueError("invalid longitudinal sample or mode weights")
    modes = longitudinal_modes(target, neutral_threshold=neutral_threshold)
    element_weights = mode_weights.to(prediction)[modes]
    error = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    combined = element_weights * sample_weights.reshape(-1, 1)
    return torch.sum(error * combined) / torch.clamp(torch.sum(combined), min=1e-8)


def weighted_longitudinal_derivative_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("longitudinal actions must be matching BxH tensors")
    if prediction.shape[1] < 2 or sample_weights.numel() != prediction.shape[0]:
        raise ValueError("longitudinal derivatives require a temporal horizon")
    error = nn.functional.smooth_l1_loss(
        prediction[:, 1:] - prediction[:, :-1],
        target[:, 1:] - target[:, :-1],
        reduction="none",
    )
    per_sample = torch.mean(error, dim=1)
    weights = sample_weights.reshape(-1)
    return torch.sum(per_sample * weights) / torch.clamp(torch.sum(weights), min=1e-8)


def weighted_mode_classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    neutral_threshold: float = 0.05,
) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[:2] != target.shape or logits.shape[2] != 3:
        raise ValueError("mode logits must be BxHx3 and align with targets")
    if sample_weights.numel() != target.shape[0] or class_weights.shape != (3,):
        raise ValueError("invalid mode-classification weights")
    modes = longitudinal_modes(target, neutral_threshold=neutral_threshold)
    losses = nn.functional.cross_entropy(
        logits.reshape(-1, 3),
        modes.reshape(-1),
        weight=class_weights.to(logits),
        reduction="none",
    ).reshape_as(target)
    per_sample = torch.mean(losses, dim=1)
    weights = sample_weights.reshape(-1)
    return torch.sum(per_sample * weights) / torch.clamp(torch.sum(weights), min=1e-8)


def set_longitudinal_only_trainable(model: FactorizedTemporalPolicy) -> None:
    """Freeze the released steering/perception path and expose longitudinal heads."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (
        model.longitudinal_trunk,
        model.longitudinal_action_head,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True
    model.set_encoder_trainable(False)


def longitudinal_finetuning_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    first_action_weight: float,
    chunk_weight: float,
    derivative_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optimize the deterministic axis against the RMSE-based promotion gate."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("longitudinal actions must be matching BxH tensors")
    if prediction.shape[1] < 2:
        raise ValueError("longitudinal fine-tuning requires a temporal horizon")
    weights = (first_action_weight, chunk_weight, derivative_weight)
    if any(weight < 0 for weight in weights) or first_action_weight <= 0:
        raise ValueError("longitudinal fine-tuning weights are invalid")
    components = {
        "first_action_mse": nn.functional.mse_loss(
            prediction[:, 0], target[:, 0]
        ),
        "chunk_mse": nn.functional.mse_loss(prediction, target),
        "derivative_mse": nn.functional.mse_loss(
            prediction[:, 1:] - prediction[:, :-1],
            target[:, 1:] - target[:, :-1],
        ),
    }
    loss = (
        first_action_weight * components["first_action_mse"]
        + chunk_weight * components["chunk_mse"]
        + derivative_weight * components["derivative_mse"]
    )
    return loss, components
