"""Temporal observation-conditioned diffusion policy for bounded action chunks."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def cosine_beta_schedule(steps: int, cosine_s: float = 0.008) -> torch.Tensor:
    if steps < 2 or not 0 <= cosine_s < 1:
        raise ValueError("invalid cosine diffusion schedule")
    timeline = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    cumulative = torch.cos(((timeline + cosine_s) / (1 + cosine_s)) * math.pi / 2) ** 2
    cumulative = cumulative / cumulative[0]
    betas = 1 - cumulative[1:] / cumulative[:-1]
    return torch.clamp(betas, min=1e-6, max=0.999).float()


class DiffusionSchedule(nn.Module):
    betas: torch.Tensor
    alphas_cumulative: torch.Tensor
    sqrt_alphas_cumulative: torch.Tensor
    sqrt_one_minus_alphas_cumulative: torch.Tensor

    def __init__(self, steps: int = 100, cosine_s: float = 0.008) -> None:
        super().__init__()
        betas = cosine_beta_schedule(steps, cosine_s)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        self.steps = steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumulative", cumulative)
        self.register_buffer("sqrt_alphas_cumulative", torch.sqrt(cumulative))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumulative", torch.sqrt(1.0 - cumulative)
        )

    @staticmethod
    def _extract(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if timesteps.ndim != 1 or timesteps.shape[0] != target.shape[0]:
            raise ValueError("diffusion timesteps must have one value per batch item")
        extracted = values.gather(0, timesteps)
        return extracted.reshape(target.shape[0], *((1,) * (target.ndim - 1)))

    def add_noise(
        self,
        clean_actions: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if clean_actions.shape != noise.shape or clean_actions.ndim != 3:
            raise ValueError("clean actions and noise must be matching BxHxD tensors")
        clean_scale = self._extract(
            self.sqrt_alphas_cumulative, timesteps, clean_actions
        )
        noise_scale = self._extract(
            self.sqrt_one_minus_alphas_cumulative, timesteps, clean_actions
        )
        return clean_scale * clean_actions + noise_scale * noise

    def predict_clean_actions(
        self,
        noisy_actions: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_actions.shape != predicted_noise.shape or noisy_actions.ndim != 3:
            raise ValueError("diffusion action tensors must be matching BxHxD values")
        clean_scale = self._extract(
            self.sqrt_alphas_cumulative, timesteps, noisy_actions
        )
        noise_scale = self._extract(
            self.sqrt_one_minus_alphas_cumulative, timesteps, noisy_actions
        )
        reconstructed = (noisy_actions - noise_scale * predicted_noise) / torch.clamp(
            clean_scale, min=1e-6
        )
        return torch.clamp(reconstructed, -1.0, 1.0)

    @torch.inference_mode()
    def ddim_sample(
        self,
        model: TemporalDiffusionPolicy,
        images: torch.Tensor,
        state_history: torch.Tensor,
        condition: torch.Tensor,
        *,
        inference_steps: int,
        eta: float = 0.0,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not 1 <= inference_steps <= self.steps or eta < 0:
            raise ValueError("invalid DDIM inference configuration")
        context = model.encode_observation(images, state_history, condition)
        batch_size = images.shape[0]
        actions = (
            torch.randn(
                batch_size,
                model.action_horizon,
                model.action_dimension,
                device=images.device,
                dtype=images.dtype,
            )
            if initial_noise is None
            else initial_noise.clone()
        )
        if actions.shape != (
            batch_size,
            model.action_horizon,
            model.action_dimension,
        ):
            raise ValueError("initial DDIM noise has the wrong shape")
        times = torch.linspace(
            self.steps - 1,
            0,
            inference_steps,
            device=images.device,
        ).round().long()
        times = torch.unique_consecutive(times)
        for index, timestep in enumerate(times):
            timestep_batch = torch.full(
                (batch_size,), int(timestep), device=images.device, dtype=torch.long
            )
            predicted_noise = model.denoise(actions, timestep_batch, context)
            alpha = self.alphas_cumulative[timestep]
            predicted_clean = (
                actions - torch.sqrt(1.0 - alpha) * predicted_noise
            ) / torch.sqrt(alpha)
            predicted_clean = torch.clamp(predicted_clean, -1.0, 1.0)
            if index == len(times) - 1:
                actions = predicted_clean
                continue
            previous_timestep = times[index + 1]
            previous_alpha = self.alphas_cumulative[previous_timestep]
            sigma = eta * torch.sqrt(
                torch.clamp(
                    (1 - previous_alpha) / (1 - alpha)
                    * (1 - alpha / previous_alpha),
                    min=0.0,
                )
            )
            direction = torch.sqrt(
                torch.clamp(1 - previous_alpha - sigma**2, min=0.0)
            ) * predicted_noise
            stochastic = sigma * torch.randn_like(actions) if eta > 0 else 0.0
            actions = torch.sqrt(previous_alpha) * predicted_clean + direction + stochastic
        return torch.clamp(actions, -1.0, 1.0)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("time embedding dimension must be even and at least four")
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)


class TemporalDiffusionPolicy(nn.Module):
    """Encode temporal observations once and denoise a complete action sequence."""

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
        dropout: float = 0.1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        dimensions = (
            history_frames,
            action_horizon,
            action_dimension,
            state_dimension,
            condition_dimension,
            image_projection_dimension,
            state_projection_dimension,
            temporal_hidden_dimension,
            condition_projection_dimension,
            denoiser_dimension,
            denoiser_layers,
            denoiser_heads,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all diffusion model dimensions must be positive")
        if denoiser_dimension % denoiser_heads or not 0 <= dropout < 1:
            raise ValueError("invalid diffusion transformer configuration")
        self.history_frames = history_frames
        self.action_horizon = action_horizon
        self.action_dimension = action_dimension
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = efficientnet_b0(weights=weights)
        encoder_dimension = self.encoder.classifier[1].in_features
        self.encoder.classifier = nn.Identity()
        self.image_projection = nn.Sequential(
            nn.Linear(encoder_dimension, image_projection_dimension),
            nn.LayerNorm(image_projection_dimension),
            nn.SiLU(),
        )
        self.state_projection = nn.Sequential(
            nn.Linear(state_dimension, state_projection_dimension),
            nn.LayerNorm(state_projection_dimension),
            nn.SiLU(),
        )
        self.temporal_encoder = nn.GRU(
            image_projection_dimension + state_projection_dimension,
            temporal_hidden_dimension,
            batch_first=True,
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dimension, condition_projection_dimension),
            nn.LayerNorm(condition_projection_dimension),
            nn.SiLU(),
        )
        context_dimension = temporal_hidden_dimension + condition_projection_dimension
        self.context_projection = nn.Linear(context_dimension, denoiser_dimension)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(denoiser_dimension),
            nn.Linear(denoiser_dimension, denoiser_dimension),
            nn.SiLU(),
            nn.Linear(denoiser_dimension, denoiser_dimension),
        )
        self.action_projection = nn.Linear(action_dimension, denoiser_dimension)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, action_horizon, denoiser_dimension)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=denoiser_dimension,
            nhead=denoiser_heads,
            dim_feedforward=denoiser_dimension * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.denoiser = nn.TransformerEncoder(
            layer, num_layers=denoiser_layers, enable_nested_tensor=False
        )
        self.noise_head = nn.Sequential(
            nn.LayerNorm(denoiser_dimension),
            nn.Linear(denoiser_dimension, action_dimension),
        )
        self._encoder_trainable = True

    def set_encoder_trainable(self, trainable: bool) -> None:
        self._encoder_trainable = trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def train(self, mode: bool = True) -> TemporalDiffusionPolicy:
        super().train(mode)
        if mode and not self._encoder_trainable:
            self.encoder.eval()
        return self

    def encode_observation(
        self,
        images: torch.Tensor,
        state_history: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 5 or state_history.ndim != 3 or condition.ndim != 2:
            raise ValueError("invalid temporal diffusion observation ranks")
        batch_size, history = images.shape[:2]
        if history != self.history_frames or state_history.shape[:2] != (batch_size, history):
            raise ValueError("diffusion observation history does not match the contract")
        flattened = images.reshape(batch_size * history, *images.shape[2:])
        image_features = self.image_projection(self.encoder(flattened)).reshape(
            batch_size, history, -1
        )
        state_features = self.state_projection(state_history)
        sequence, _ = self.temporal_encoder(
            torch.cat((image_features, state_features), dim=2)
        )
        condition_features = self.condition_projection(condition)
        return torch.cat((sequence[:, -1], condition_features), dim=1)

    def denoise(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_actions.shape[1:] != (self.action_horizon, self.action_dimension):
            raise ValueError("noisy action sequence does not match the model contract")
        if (
            timesteps.shape != (noisy_actions.shape[0],)
            or context.shape[0] != noisy_actions.shape[0]
        ):
            raise ValueError("diffusion conditioning batch dimensions do not match")
        conditioning = self.context_projection(context) + self.time_embedding(timesteps)
        tokens = (
            self.action_projection(noisy_actions)
            + self.position_embedding
            + conditioning.unsqueeze(1)
        )
        return cast(torch.Tensor, self.noise_head(self.denoiser(tokens)))

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        state_history: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_observation(images, state_history, condition)
        return self.denoise(noisy_actions, timesteps, context)


def weighted_noise_mse(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("diffusion noise tensors must be matching BxHxD tensors")
    if weights.numel() != prediction.shape[0]:
        raise ValueError("one diffusion sample weight is required per batch item")
    per_sample = torch.mean(torch.square(prediction - target), dim=(1, 2))
    flat_weights = weights.reshape(-1)
    return torch.sum(per_sample * flat_weights) / torch.clamp(
        torch.sum(flat_weights), min=1e-8
    )


def weighted_action_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    longitudinal_weight: float = 2.0,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("reconstructed actions must be matching BxHxD tensors")
    if prediction.shape[2] != 2 or sample_weights.numel() != prediction.shape[0]:
        raise ValueError("invalid action dimensions or sample weights")
    if longitudinal_weight < 1:
        raise ValueError("longitudinal reconstruction weight must be at least one")
    dimension_weights = prediction.new_tensor([1.0, longitudinal_weight])
    error = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    per_sample = torch.mean(error * dimension_weights, dim=(1, 2))
    weights = sample_weights.reshape(-1)
    return torch.sum(per_sample * weights) / torch.clamp(torch.sum(weights), min=1e-8)


def weighted_temporal_derivative_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    longitudinal_weight: float = 2.0,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("temporal action tensors must be matching BxHxD values")
    if prediction.shape[1] < 2:
        raise ValueError("temporal derivative loss requires at least two actions")
    return weighted_action_reconstruction_loss(
        prediction[:, 1:] - prediction[:, :-1],
        target[:, 1:] - target[:, :-1],
        sample_weights,
        longitudinal_weight=longitudinal_weight,
    )
