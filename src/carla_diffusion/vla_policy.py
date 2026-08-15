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
