"""Four-frame behavioral-cloning model for the temporal-context ablation."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class TemporalBC(nn.Module):
    """Encode four image/state tokens with a shared backbone and GRU."""

    def __init__(
        self,
        *,
        history_frames: int = 4,
        state_dimension: int = 10,
        condition_dimension: int = 9,
        image_projection_dimension: int = 256,
        state_projection_dimension: int = 64,
        temporal_hidden_dimension: int = 256,
        condition_projection_dimension: int = 64,
        temporal_layers: int = 1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        dimensions = (
            history_frames,
            state_dimension,
            condition_dimension,
            image_projection_dimension,
            state_projection_dimension,
            temporal_hidden_dimension,
            condition_projection_dimension,
            temporal_layers,
        )
        if any(dimension <= 0 for dimension in dimensions):
            raise ValueError("all temporal model dimensions must be positive")
        self.history_frames = history_frames
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
            input_size=image_projection_dimension + state_projection_dimension,
            hidden_size=temporal_hidden_dimension,
            num_layers=temporal_layers,
            batch_first=True,
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_dimension, condition_projection_dimension),
            nn.LayerNorm(condition_projection_dimension),
            nn.SiLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(temporal_hidden_dimension + condition_projection_dimension, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
            nn.Tanh(),
        )
        self._encoder_trainable = True

    def set_encoder_trainable(self, trainable: bool) -> None:
        self._encoder_trainable = trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def train(self, mode: bool = True) -> TemporalBC:
        super().train(mode)
        if mode and not self._encoder_trainable:
            self.encoder.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        state_history: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if images.ndim != 5 or state_history.ndim != 3:
            raise ValueError("temporal inputs must be BxTxCxHxW images and BxTxD states")
        batch_size, history = images.shape[:2]
        if history != self.history_frames or state_history.shape[1] != history:
            raise ValueError("temporal history length does not match the model contract")
        flattened_images = images.reshape(
            batch_size * history,
            images.shape[2],
            images.shape[3],
            images.shape[4],
        )
        image_features = self.encoder(flattened_images)
        image_tokens = self.image_projection(image_features).reshape(batch_size, history, -1)
        state_tokens = self.state_projection(state_history)
        tokens = torch.cat((image_tokens, state_tokens), dim=2)
        sequence, _ = self.temporal_encoder(tokens)
        condition_features = self.condition_projection(condition)
        output = self.action_head(torch.cat((sequence[:, -1], condition_features), dim=1))
        if not isinstance(output, torch.Tensor):
            raise TypeError("temporal action head must return a tensor")
        return output
