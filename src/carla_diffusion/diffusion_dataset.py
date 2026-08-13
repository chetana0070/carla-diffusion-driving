"""Action-chunk dataset view for the temporal diffusion policy."""

from __future__ import annotations

from pathlib import Path

import torch

from .temporal_bc_dataset import TemporalWindowDataset


class DiffusionWindowDataset(TemporalWindowDataset):
    """Return the complete future action horizon from a prepared temporal window."""

    def __init__(
        self,
        processed_root: str | Path,
        split: str,
        *,
        action_horizon: int = 16,
        history_frames: int = 4,
        image_size: int = 224,
        augment: bool = False,
        normalized_state_clip: float = 10.0,
        required_category: str | None = None,
    ) -> None:
        if action_horizon < 2:
            raise ValueError("diffusion action horizon must contain at least two actions")
        super().__init__(
            processed_root,
            split,
            history_frames=history_frames,
            image_size=image_size,
            augment=augment,
            normalized_state_clip=normalized_state_clip,
            required_category=required_category,
        )
        if any(len(row["action_target"]) != action_horizon for row in self.rows):
            raise ValueError("prepared action targets do not match the diffusion horizon")
        self.action_horizon = action_horizon

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        target = torch.tensor(self.rows[index]["action_target"], dtype=torch.float32)
        if target.shape != (self.action_horizon, 2):
            raise ValueError(f"invalid action chunk at dataset index {index}")
        item["target"] = target
        return item
