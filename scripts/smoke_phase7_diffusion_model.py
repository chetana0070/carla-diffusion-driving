#!/usr/bin/env python3
"""Run a dependency-free-of-dataset Phase 7 model and DDIM contract smoke."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_diffusion_policy import build_model

from carla_diffusion.config import load_and_validate_config
from carla_diffusion.diffusion_policy import DiffusionSchedule, TemporalDiffusionPolicy


def main() -> int:
    project_config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    config = project_config["diffusion_policy"]
    model: TemporalDiffusionPolicy = build_model(config, pretrained=False).eval()
    schedule = DiffusionSchedule(
        int(config["diffusion_steps"]), float(config["cosine_s"])
    )
    batch_size = 1
    images = torch.zeros(
        batch_size,
        int(config["history_frames"]),
        3,
        int(config["image_size"]),
        int(config["image_size"]),
    )
    states = torch.zeros(
        batch_size,
        int(config["history_frames"]),
        int(config["state_input_dimension"]),
    )
    condition = torch.zeros(batch_size, int(config["condition_input_dimension"]))
    initial_noise = torch.zeros(
        batch_size,
        int(config["action_horizon"]),
        int(config["action_dimension"]),
    )
    with torch.inference_mode():
        actions = schedule.ddim_sample(
            model,
            images,
            states,
            condition,
            inference_steps=int(config["inference_steps"]),
            initial_noise=initial_noise,
        )
    passed = actions.shape == initial_noise.shape and bool(torch.all(torch.isfinite(actions)))
    report = {
        "status": "passed" if passed else "failed",
        "model_type": "temporal_diffusion_policy",
        "action_shape": list(actions.shape),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "bounded": bool(torch.all(torch.abs(actions) <= 1.0)),
        "diffusion_steps": schedule.steps,
        "inference_steps": int(config["inference_steps"]),
    }
    print(json.dumps(report, indent=2))
    return 0 if passed and report["bounded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
