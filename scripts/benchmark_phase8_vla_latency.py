#!/usr/bin/env python3
"""Measure deterministic batch-one hierarchical VLA planner latency on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.closed_loop import latency_summary
from carla_diffusion.vla_policy import HierarchicalVLAPlanner
from carla_diffusion.vla_smoothing import apply_checkpoint_smoother


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts/checkpoints/phase8_vla_v220/best.pt",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(checkpoint: dict[str, Any]) -> HierarchicalVLAPlanner:
    config = checkpoint["config"]["vision_language_action"]
    return HierarchicalVLAPlanner(
        vocabulary_size=len(checkpoint["vocabulary"]),
        state_dimension=int(config["state_input_dimension"]),
        action_horizon=int(config["action_horizon"]),
        language_dimension=int(config["language_dimension"]),
        state_hidden_dimension=int(config["state_hidden_dimension"]),
        fusion_dimension=int(config["fusion_dimension"]),
        pretrained_visual_encoder=False,
    )


def main() -> int:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase 8 latency benchmark")
    device = torch.device("cuda")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get("model_type") not in {
        "hierarchical_vla_preflight",
        "hierarchical_vla_smoothed",
    }:
        raise ValueError("checkpoint is not a hierarchical VLA planner")
    config = checkpoint["config"]["vision_language_action"]
    model = build_model(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
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
    token_ids = torch.zeros(
        1, int(config["max_instruction_tokens"]), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(token_ids, dtype=torch.float32)
    token_ids[:, :2] = torch.tensor([2, 3], device=device)
    attention_mask[:, :2] = 1.0

    @torch.inference_mode()
    def predict() -> torch.Tensor:
        raw = model(images, states, token_ids, attention_mask)
        return apply_checkpoint_smoother(raw, checkpoint)

    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.warmup):
        predict()
    torch.cuda.synchronize()
    latencies: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        prediction = predict()
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
    if not bool(torch.all(torch.isfinite(prediction))):
        raise FloatingPointError("VLA latency benchmark produced non-finite actions")
    summary = latency_summary(latencies)
    budget = float(config["maximum_planner_latency_ms"])
    report = {
        "status": "passed",
        "scope": "normalized_tensors_to_complete_action_chunk",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": torch.cuda.get_device_name(0),
        "batch_size": 1,
        "history_frames": int(config["history_frames"]),
        "action_horizon": int(config["action_horizon"]),
        "deployment_transform": checkpoint.get("deployment_transform"),
        "latency": summary,
        "latency_budget_ms": budget,
        "p95_latency_gate_passed": summary["p95_ms"] <= budget,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["p95_latency_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
