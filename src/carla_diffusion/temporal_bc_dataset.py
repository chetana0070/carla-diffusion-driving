"""Temporal PyTorch dataset backed by the Phase 3 four-frame windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .bc_dataset import IMAGENET_MEAN, IMAGENET_STD


class TemporalWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        processed_root: str | Path,
        split: str,
        *,
        history_frames: int = 4,
        image_size: int = 224,
        augment: bool = False,
        normalized_state_clip: float = 10.0,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split: {split}")
        if history_frames <= 1 or image_size <= 0 or normalized_state_clip <= 0:
            raise ValueError("invalid temporal dataset dimensions")
        root = Path(processed_root).resolve()
        report = _load_json(root / "report.json")
        normalization = _load_json(root / "normalization.json")
        self.dataset_root = Path(str(report["source_dataset"])).resolve()
        all_rows = [
            json.loads(line)
            for line in (root / "windows.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.rows = [row for row in all_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"no windows found for split: {split}")
        if any(len(row["image_paths"]) != history_frames for row in self.rows):
            raise ValueError("window image history does not match the temporal contract")
        self.state_mean = torch.tensor(normalization["state_mean"], dtype=torch.float32)
        self.state_std = torch.tensor(normalization["state_std"], dtype=torch.float32)
        self.normalized_state_clip = normalized_state_clip
        self.to_float = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        transforms: list[Any] = [v2.Resize((image_size, image_size), antialias=True)]
        if augment:
            # Called once on TxCxHxW so all frames share one photometric transform.
            transforms.append(v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1))
        transforms.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.sequence_transform = v2.Compose(transforms)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        frames = []
        for relative_path in row["image_paths"]:
            with Image.open(self.dataset_root / str(relative_path)) as image_file:
                frames.append(self.to_float(image_file.convert("RGB")))
        images = self.sequence_transform(torch.stack(frames))
        state_history = torch.tensor(row["state_history"], dtype=torch.float32)
        normalized_states = (state_history - self.state_mean) / self.state_std
        if not bool(torch.all(torch.isfinite(normalized_states))):
            raise ValueError(f"non-finite temporal state at dataset index {index}")
        normalized_states = torch.clamp(
            normalized_states,
            min=-self.normalized_state_clip,
            max=self.normalized_state_clip,
        )
        return {
            "images": images,
            "state_history": normalized_states,
            "condition": torch.tensor(row["condition"], dtype=torch.float32),
            "target": torch.tensor(row["action_target"][0], dtype=torch.float32),
            "weight": torch.tensor(float(row["sample_weight"]), dtype=torch.float32),
        }


def _load_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload
