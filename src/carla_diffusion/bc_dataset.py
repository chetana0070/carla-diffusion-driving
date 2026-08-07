"""PyTorch dataset backed by the Phase 3 temporal-window index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SingleFrameWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Expose the current frame/action while preserving route-level splits."""

    def __init__(
        self,
        processed_root: str | Path,
        split: str,
        *,
        image_size: int = 224,
        augment: bool = False,
        normalized_state_clip: float = 10.0,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split: {split}")
        if normalized_state_clip <= 0:
            raise ValueError("normalized-state clip must be positive")
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
        self.state_mean = torch.tensor(normalization["state_mean"], dtype=torch.float32)
        self.state_std = torch.tensor(normalization["state_std"], dtype=torch.float32)
        self.normalized_state_clip = normalized_state_clip
        transforms: list[Any] = [v2.Resize((image_size, image_size), antialias=True)]
        if augment:
            # Geometry is intentionally unchanged: flips alter left/right driving semantics.
            transforms.append(v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1))
        transforms.extend(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.image_transform = v2.Compose(transforms)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image_path = self.dataset_root / str(row["image_paths"][-1])
        with Image.open(image_path) as image_file:
            image = self.image_transform(image_file.convert("RGB"))
        state = torch.tensor(row["state_history"][-1], dtype=torch.float32)
        normalized_state = (state - self.state_mean) / self.state_std
        if not bool(torch.all(torch.isfinite(normalized_state))):
            raise ValueError(f"non-finite normalized state at dataset index {index}")
        normalized_state = torch.clamp(
            normalized_state,
            min=-self.normalized_state_clip,
            max=self.normalized_state_clip,
        )
        condition = torch.tensor(row["condition"], dtype=torch.float32)
        return {
            "image": image,
            "scalar_context": torch.cat((normalized_state, condition)),
            "target": torch.tensor(row["action_target"][0], dtype=torch.float32),
            "weight": torch.tensor(float(row["sample_weight"]), dtype=torch.float32),
        }


def _load_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload
