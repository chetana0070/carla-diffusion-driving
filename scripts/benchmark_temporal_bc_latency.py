#!/usr/bin/env python3
"""Measure batch-one temporal model inference latency on CUDA."""

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

from carla_diffusion.closed_loop import latency_summary
from carla_diffusion.temporal_bc_model import TemporalBC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / "phase5_temporal_bc_v080"
            / "best.pt"
        ),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    return parser.parse_args()


def build_model(config: dict[str, Any]) -> TemporalBC:
    return TemporalBC(
        history_frames=int(config["history_frames"]),
        state_dimension=int(config["state_input_dimension"]),
        condition_dimension=int(config["condition_input_dimension"]),
        image_projection_dimension=int(config["image_projection_dimension"]),
        state_projection_dimension=int(config["state_projection_dimension"]),
        temporal_hidden_dimension=int(config["temporal_hidden_dimension"]),
        condition_projection_dimension=int(config["condition_projection_dimension"]),
        temporal_layers=int(config["temporal_layers"]),
        pretrained=False,
    )


def main() -> int:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the latency benchmark")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    project_config = checkpoint["config"]
    temporal = project_config["temporal_behavioral_cloning"]
    model = build_model(temporal).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    images = torch.zeros(
        1,
        int(temporal["history_frames"]),
        3,
        int(temporal["image_size"]),
        int(temporal["image_size"]),
        device=device,
    )
    states = torch.zeros(
        1,
        int(temporal["history_frames"]),
        int(temporal["state_input_dimension"]),
        device=device,
    )
    condition = torch.zeros(
        1, int(temporal["condition_input_dimension"]), device=device
    )
    latencies: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(images, states, condition)
        torch.cuda.synchronize()
        for _ in range(args.iterations):
            started = time.perf_counter()
            prediction = model(images, states, condition)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)
    if not bool(torch.all(torch.isfinite(prediction))):
        raise FloatingPointError("benchmark produced a non-finite prediction")
    summary = latency_summary(latencies)
    budget = float(project_config["evaluation"]["max_policy_latency_ms"])
    report = {
        "status": "passed",
        "scope": "model_inference_only",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": torch.cuda.get_device_name(0),
        "batch_size": 1,
        "history_frames": int(temporal["history_frames"]),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "latency": summary,
        "latency_budget_ms": budget,
        "p95_latency_gate_passed": summary["p95_ms"] <= budget,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
