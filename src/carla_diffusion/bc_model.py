"""Single-frame behavioral-cloning model used as the first learned baseline."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class SingleFrameBC(nn.Module):
    """Fuse one RGB frame with current normalized driving context."""

    def __init__(
        self,
        *,
        scalar_dimension: int = 19,
        scalar_feature_dimension: int = 128,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if scalar_dimension <= 0 or scalar_feature_dimension <= 0:
            raise ValueError("feature dimensions must be positive")
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = efficientnet_b0(weights=weights)
        image_feature_dimension = self.encoder.classifier[1].in_features
        self.encoder.classifier = nn.Identity()
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dimension, scalar_feature_dimension),
            nn.LayerNorm(scalar_feature_dimension),
            nn.SiLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(image_feature_dimension + scalar_feature_dimension, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
            nn.Tanh(),
        )
        self._encoder_trainable = True

    def set_encoder_trainable(self, trainable: bool) -> None:
        self._encoder_trainable = trainable
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def train(self, mode: bool = True) -> SingleFrameBC:
        super().train(mode)
        # A frozen backbone must not update BatchNorm running statistics.
        if mode and not self._encoder_trainable:
            self.encoder.eval()
        return self

    def forward(self, image: torch.Tensor, scalar_context: torch.Tensor) -> torch.Tensor:
        image_features = self.encoder(image)
        scalar_features = self.scalar_encoder(scalar_context)
        output = self.action_head(torch.cat((image_features, scalar_features), dim=1))
        if not isinstance(output, torch.Tensor):
            raise TypeError("action head must return a tensor")
        return output


def weighted_action_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Mean two-action MSE per item, normalized by the sum of sample weights."""
    per_sample = torch.mean(torch.square(prediction - target), dim=1)
    weights = sample_weight.reshape(-1).to(dtype=per_sample.dtype)
    return torch.sum(per_sample * weights) / torch.clamp(torch.sum(weights), min=1e-8)
