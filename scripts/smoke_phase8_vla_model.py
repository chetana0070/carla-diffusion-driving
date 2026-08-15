#!/usr/bin/env python3
"""Run a synthetic forward/backward pass through the Phase 8 VLA planner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from carla_diffusion.config import load_and_validate_config
from carla_diffusion.vla_data import build_vocabulary
from carla_diffusion.vla_policy import HierarchicalVLAPlanner, weighted_action_chunk_loss


def main() -> int:
    config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    vla = config["vision_language_action"]
    vocabulary = build_vocabulary()
    model = HierarchicalVLAPlanner(
        vocabulary_size=len(vocabulary),
        state_dimension=int(vla["state_input_dimension"]),
        action_horizon=int(vla["action_horizon"]),
        language_dimension=int(vla["language_dimension"]),
        state_hidden_dimension=int(vla["state_hidden_dimension"]),
        fusion_dimension=int(vla["fusion_dimension"]),
    )
    batch = 2
    history = int(vla["history_frames"])
    image_size = int(vla["image_size"])
    token_count = int(vla["max_instruction_tokens"])
    target = torch.zeros(batch, int(vla["action_horizon"]), 2)
    prediction = model(
        torch.rand(batch, history, 3, image_size, image_size),
        torch.rand(batch, history, int(vla["state_input_dimension"])),
        torch.ones(batch, token_count, dtype=torch.long),
        torch.ones(batch, token_count),
    )
    loss = weighted_action_chunk_loss(prediction, target, torch.ones(batch))
    loss.backward()
    payload = {
        "status": "passed",
        "prediction_shape": list(prediction.shape),
        "bounded": bool(torch.all(torch.abs(prediction) <= 1.0)),
        "loss": float(loss.detach()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
