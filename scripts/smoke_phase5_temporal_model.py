#!/usr/bin/env python3
"""CUDA shape, gradient, range, and memory smoke test for temporal BC."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.bc_model import weighted_action_mse
from carla_diffusion.temporal_bc_model import TemporalBC


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the temporal model smoke test")
    device = torch.device("cuda")
    model = TemporalBC(pretrained=False).to(device)
    images = torch.randn(2, 4, 3, 224, 224, device=device)
    states = torch.randn(2, 4, 10, device=device)
    condition = torch.randn(2, 9, device=device)
    target = torch.empty(2, 2, device=device).uniform_(-1, 1)
    weight = torch.tensor([0.25, 2.0], device=device)
    prediction = model(images, states, condition)
    loss = weighted_action_mse(prediction, target, weight)
    loss.backward()
    result = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "image_shape": list(images.shape),
        "state_shape": list(states.shape),
        "prediction_shape": list(prediction.shape),
        "prediction_in_range": bool(torch.all(torch.abs(prediction) <= 1.0)),
        "loss": float(loss.detach()),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
