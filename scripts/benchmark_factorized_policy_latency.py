#!/usr/bin/env python3
"""Measure complete batch-one factorized policy latency on CUDA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_factorized_policy import build_model

from carla_diffusion.closed_loop import latency_summary
from carla_diffusion.diffusion_policy import DiffusionSchedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / "phase7_factorized_v190"
            / "best.pt"
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the factorized latency benchmark")
    device = torch.device("cuda")
    checkpoint: dict[str, Any] = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    if checkpoint.get("model_type") != "factorized_temporal_policy":
        raise ValueError("checkpoint is not a factorized temporal policy")
    project_config = checkpoint["config"]
    config = project_config["factorized_policy"]
    model = build_model(config, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    schedule = DiffusionSchedule(
        int(config["diffusion_steps"]), float(config["cosine_s"])
    ).to(device)
    images = torch.zeros(
        1,
        int(config["history_frames"]),
        3,
        int(config["image_size"]),
        int(config["image_size"]),
        device=device,
    )
    states = torch.zeros(
        1,
        int(config["history_frames"]),
        int(config["state_input_dimension"]),
        device=device,
    )
    condition = torch.zeros(
        1, int(config["condition_input_dimension"]), device=device
    )
    initial_noise = torch.zeros(
        1,
        int(config["action_horizon"]),
        int(config["action_dimension"]),
        device=device,
    )

    def sample() -> torch.Tensor:
        return model.sample(
            schedule,
            images,
            states,
            condition,
            inference_steps=int(config["inference_steps"]),
            eta=float(config["ddim_eta"]),
            initial_noise=initial_noise,
        )

    latencies: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        sample()
    torch.cuda.synchronize()
    for _ in range(args.iterations):
        started = time.perf_counter()
        prediction = sample()
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
    if not bool(torch.all(torch.isfinite(prediction))):
        raise FloatingPointError("factorized benchmark produced non-finite actions")
    summary = latency_summary(latencies)
    budget = float(project_config["evaluation"]["max_policy_latency_ms"])
    report = {
        "status": "passed",
        "scope": "shared_observation_encoding_plus_steering_ddim_plus_longitudinal_head",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": torch.cuda.get_device_name(0),
        "batch_size": 1,
        "history_frames": int(config["history_frames"]),
        "action_horizon": int(config["action_horizon"]),
        "inference_steps": int(config["inference_steps"]),
        "latency": summary,
        "latency_budget_ms": budget,
        "p95_latency_gate_passed": summary["p95_ms"] <= budget,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["p95_latency_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
