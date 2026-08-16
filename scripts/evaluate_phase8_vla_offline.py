#!/usr/bin/env python3
"""Run the deterministic Phase 8.1 offline VLA promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_phase8_vla_latency import build_model

from carla_diffusion.config import load_and_validate_config
from carla_diffusion.diffusion_evaluation import (
    summarize_action_pairs,
    summarize_chunk_smoothness,
)
from carla_diffusion.vla_dataset import VLAWindowDataset
from carla_diffusion.vla_evaluation import (
    summarize_instruction_slices,
    vla_promotion_decision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts/checkpoints/phase8_vla_v220/best.pt",
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts/checkpoints/phase8_vla_v220/report.json",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/evaluations/phase7_factorized_longitudinal_gate_v200.json"
        ),
    )
    parser.add_argument(
        "--latency-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts/evaluations/phase8_vla_latency_v230.json",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data/processed/phase8_vla_v1",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data/raw/phase2_pilot_v2_v043",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/evaluations/phase8_vla_offline_gate_v230.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def baseline_metrics(report: dict[str, Any]) -> dict[str, Any]:
    candidate = report.get("factorized_test")
    if isinstance(candidate, dict) and isinstance(candidate.get("first_action"), dict):
        return candidate["first_action"]
    for key in ("candidate_test", "test_metrics", "first_action"):
        value = report.get(key)
        if isinstance(value, dict) and "joint_rmse" in value:
            return value
    raise ValueError("baseline report does not contain first-action metrics")


def read_latency_gate(path: Path, checkpoint_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        return {"passed": False, "reason": "latency report is missing"}
    report = json.loads(path.read_text(encoding="utf-8"))
    same_checkpoint = report.get("checkpoint_sha256") == checkpoint_sha256
    return {
        "passed": bool(report.get("p95_latency_gate_passed")) and same_checkpoint,
        "same_checkpoint": same_checkpoint,
        "p95_ms": report.get("latency", {}).get("p95_ms"),
        "budget_ms": report.get("latency_budget_ms"),
        "report": str(path),
    }


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch size must be positive and workers cannot be negative")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required; pass --allow-cpu only for diagnostics")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha256 = sha256(checkpoint_path)
    checkpoint: dict[str, Any] = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get("model_type") != "hierarchical_vla_preflight":
        raise ValueError("candidate checkpoint has the wrong model type")
    training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    checkpoint_integrity = (
        training_report.get("status") == "passed"
        and training_report.get("checkpoint_sha256") == checkpoint_sha256
        and training_report.get("best_epoch") == checkpoint.get("epoch")
    )
    baseline = baseline_metrics(baseline_report)
    baseline_provenance = bool(
        baseline_report.get("promotion_gate_passed")
        or baseline_report.get("gate_passed")
    )
    current_config = load_and_validate_config(PROJECT_ROOT / "configs/project.json")
    thresholds = current_config["vla_offline_evaluation"]
    config = checkpoint["config"]["vision_language_action"]
    dataset = VLAWindowDataset(
        args.processed_root.resolve(),
        "test",
        history_frames=int(config["history_frames"]),
        planner_horizon=int(config["action_horizon"]),
        image_size=int(config["image_size"]),
        normalized_state_clip=float(config["normalized_state_clip"]),
        dataset_root=args.dataset_root.resolve(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_model(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    predicted_chunks: list[list[list[float]]] = []
    target_chunks: list[list[list[float]]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for host_batch in loader:
            batch = {
                name: tensor.to(device, non_blocking=True)
                for name, tensor in host_batch.items()
            }
            prediction = model(
                batch["images"],
                batch["state_history"],
                batch["instruction_token_ids"],
                batch["instruction_attention_mask"],
            )
            predicted_chunks.extend(prediction.float().cpu().tolist())
            target_chunks.extend(batch["target"].float().cpu().tolist())
    first_predictions = [chunk[0] for chunk in predicted_chunks]
    first_targets = [chunk[0] for chunk in target_chunks]
    flat_predictions = [action for chunk in predicted_chunks for action in chunk]
    flat_targets = [action for chunk in target_chunks for action in chunk]
    first_action = summarize_action_pairs(first_predictions, first_targets)
    full_chunk = summarize_action_pairs(flat_predictions, flat_targets)
    predicted_smoothness = summarize_chunk_smoothness(predicted_chunks)
    target_smoothness = summarize_chunk_smoothness(target_chunks)
    route_commands = [str(row["route_command"]) for row in dataset.rows]
    traffic_lights = [str(row["traffic_light_state"]) for row in dataset.rows]
    slices = summarize_instruction_slices(
        first_predictions,
        first_targets,
        route_commands,
        traffic_lights,
    )
    slice_coverage = (
        sum(value["samples"] for value in slices["route_command"].values())
        == len(dataset)
        and sum(
            value["samples"] for value in slices["traffic_light_state"].values()
        )
        == len(dataset)
    )
    latency = read_latency_gate(args.latency_report.resolve(), checkpoint_sha256)
    validation_samples = int(training_report["dataset_sizes"]["validation"])
    promotion = vla_promotion_decision(
        first_action,
        baseline,
        predicted_smoothness=predicted_smoothness,
        target_smoothness=target_smoothness,
        expected_validation_samples=int(thresholds["expected_validation_samples"]),
        validation_samples=validation_samples,
        expected_test_samples=int(thresholds["expected_test_samples"]),
        maximum_relative_rmse=float(thresholds["maximum_relative_rmse"]),
        maximum_absolute_longitudinal_bias=float(
            thresholds["maximum_absolute_longitudinal_bias"]
        ),
        maximum_longitudinal_behavior_drift=float(
            thresholds["maximum_longitudinal_behavior_drift"]
        ),
        maximum_chunk_smoothness_ratio=float(
            thresholds["maximum_chunk_smoothness_ratio"]
        ),
        latency_gate_passed=bool(latency["passed"]),
        checkpoint_integrity_passed=checkpoint_integrity,
        baseline_provenance_passed=baseline_provenance,
        instruction_slice_coverage_passed=slice_coverage,
    )
    report = {
        "status": "passed",
        "model_type": "hierarchical_vla_preflight",
        "promotion_gate_passed": promotion["gate_passed"],
        "claim_boundary": (
            "Closed-vocabulary deterministic instructions derived from route and signal "
            "metadata; no open-vocabulary or closed-loop claim."
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "baseline_report": str(args.baseline_report.resolve()),
        "device": str(device),
        "dataset_sizes": {
            "validation": validation_samples,
            "test": len(dataset),
        },
        "first_action": first_action,
        "full_action_chunk": full_chunk,
        "predicted_chunk_smoothness": predicted_smoothness,
        "target_chunk_smoothness": target_smoothness,
        "instruction_slices": slices,
        "latency_gate": latency,
        "promotion": promotion,
        "thresholds": thresholds,
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2))
    return 0 if promotion["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
