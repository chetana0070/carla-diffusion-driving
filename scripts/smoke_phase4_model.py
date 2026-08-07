#!/usr/bin/env python3
"""Fast shape, gradient, and CUDA contract check for the Phase 4 model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.bc_model import SingleFrameBC, weighted_action_mse


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase 4 model smoke test")
    device = torch.device("cuda")
    model = SingleFrameBC(pretrained=False).to(device)
    image = torch.randn(2, 3, 224, 224, device=device)
    scalar = torch.randn(2, 19, device=device)
    target = torch.empty(2, 2, device=device).uniform_(-1, 1)
    weight = torch.tensor([0.25, 2.0], device=device)
    prediction = model(image, scalar)
    loss = weighted_action_mse(prediction, target, weight)
    loss.backward()
    result = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "prediction_shape": list(prediction.shape),
        "prediction_in_range": bool(torch.all(torch.abs(prediction) <= 1.0)),
        "loss": float(loss.detach()),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
