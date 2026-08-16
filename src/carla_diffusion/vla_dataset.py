"""PyTorch dataset for deterministic Phase 8 language-conditioned windows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .bc_dataset import IMAGENET_MEAN, IMAGENET_STD


class VLAWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        processed_root: str | Path,
        split: str,
        *,
        history_frames: int,
        planner_horizon: int,
        image_size: int,
        normalized_state_clip: float,
        augment: bool = False,
        dataset_root: str | Path | None = None,
        route_command_weights: Mapping[str, float] | None = None,
        traffic_light_weights: Mapping[str, float] | None = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split: {split}")
        root = Path(processed_root).resolve()
        report = _load_json(root / "report.json")
        normalization = _load_json(root / "normalization.json")
        self.dataset_root = (
            Path(dataset_root).resolve()
            if dataset_root is not None
            else Path(str(report["source_dataset"])).resolve()
        )
        self.dataset_root_override = dataset_root is not None
        rows = [
            json.loads(line)
            for line in (root / "vla_windows.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"no VLA windows found for split: {split}")
        if any(len(row["image_paths"]) != history_frames for row in self.rows):
            raise ValueError("VLA image history does not match configuration")
        if any(len(row["action_chunk_target"]) != planner_horizon for row in self.rows):
            raise ValueError("VLA target horizon does not match configuration")
        self.state_mean = torch.tensor(normalization["state_mean"], dtype=torch.float32)
        self.state_std = torch.tensor(normalization["state_std"], dtype=torch.float32)
        self.normalized_state_clip = normalized_state_clip
        self.route_command_weights = dict(route_command_weights or {})
        self.traffic_light_weights = dict(traffic_light_weights or {})
        if any(weight <= 0 for weight in self.route_command_weights.values()):
            raise ValueError("route-command weights must be positive")
        if any(weight <= 0 for weight in self.traffic_light_weights.values()):
            raise ValueError("traffic-light weights must be positive")
        self.to_float = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        transforms: list[Any] = [v2.Resize((image_size, image_size), antialias=True)]
        if augment:
            transforms.append(v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1))
        transforms.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.sequence_transform = v2.Compose(transforms)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        source_root = self.dataset_root
        if not self.dataset_root_override and row.get("source_dataset") is not None:
            source_root = Path(str(row["source_dataset"])).resolve()
        frames = []
        for relative_path in row["image_paths"]:
            with Image.open(source_root / str(relative_path)) as image_file:
                frames.append(self.to_float(image_file.convert("RGB")))
        images = self.sequence_transform(torch.stack(frames))
        raw_states = torch.tensor(row["state_history"], dtype=torch.float32)
        states = torch.clamp(
            (raw_states - self.state_mean) / self.state_std,
            min=-self.normalized_state_clip,
            max=self.normalized_state_clip,
        )
        if not bool(torch.all(torch.isfinite(states))):
            raise ValueError(f"non-finite VLA state at dataset index {index}")
        condition_weight = self.route_command_weights.get(
            str(row["route_command"]), 1.0
        ) * self.traffic_light_weights.get(str(row["traffic_light_state"]), 1.0)
        return {
            "images": images,
            "state_history": states,
            "instruction_token_ids": torch.tensor(
                row["instruction_token_ids"], dtype=torch.long
            ),
            "instruction_attention_mask": torch.tensor(
                row["instruction_attention_mask"], dtype=torch.float32
            ),
            "target": torch.tensor(row["action_chunk_target"], dtype=torch.float32),
            "weight": torch.tensor(float(row["sample_weight"]), dtype=torch.float32),
            "condition_weight": torch.tensor(condition_weight, dtype=torch.float32),
        }


def _load_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload
