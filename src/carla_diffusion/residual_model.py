"""Bounded supervised residual for geometric recovery states."""

from __future__ import annotations

import torch
from torch import nn


class ResidualCorrection(nn.Module):
    def __init__(
        self,
        *,
        history_frames: int = 4,
        state_dimension: int = 10,
        condition_dimension: int = 9,
        hidden_dimensions: tuple[int, int] = (128, 64),
        maximum_steering_delta: float = 0.5,
        maximum_longitudinal_delta: float = 0.5,
    ) -> None:
        super().__init__()
        if min(history_frames, state_dimension, condition_dimension, *hidden_dimensions) <= 0:
            raise ValueError("residual dimensions must be positive")
        if not 0 < maximum_steering_delta <= 1:
            raise ValueError("maximum steering delta must be in (0, 1]")
        if not 0 < maximum_longitudinal_delta <= 1:
            raise ValueError("maximum longitudinal delta must be in (0, 1]")
        input_dimension = history_frames * state_dimension + condition_dimension + 2
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimensions[0]),
            nn.LayerNorm(hidden_dimensions[0]),
            nn.SiLU(),
            nn.Linear(hidden_dimensions[0], hidden_dimensions[1]),
            nn.SiLU(),
            nn.Linear(hidden_dimensions[1], 2),
            nn.Tanh(),
        )
        final_linear = self.network[5]
        if not isinstance(final_linear, nn.Linear):
            raise TypeError("residual output layer must be linear")
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        self.register_buffer(
            "maximum_delta",
            torch.tensor(
                [maximum_steering_delta, maximum_longitudinal_delta],
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        state_history: torch.Tensor,
        condition: torch.Tensor,
        base_action: torch.Tensor,
    ) -> torch.Tensor:
        if state_history.ndim != 3 or condition.ndim != 2 or base_action.ndim != 2:
            raise ValueError("residual inputs must be batched tensors")
        features = torch.cat(
            (state_history.flatten(start_dim=1), condition, base_action), dim=1
        )
        output = self.network(features) * self.maximum_delta
        if not isinstance(output, torch.Tensor):
            raise TypeError("residual network must return a tensor")
        return output


def corrected_action(base_action: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    if base_action.shape != residual.shape:
        raise ValueError("base and residual actions must have matching shapes")
    return torch.clamp(base_action + residual, min=-1.0, max=1.0)
