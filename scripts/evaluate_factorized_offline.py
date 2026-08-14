#!/usr/bin/env python3
"""Run the full offline promotion gate for the factorized temporal policy."""

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

from train_factorized_policy import build_model

from carla_diffusion.config import load_and_validate_config
from carla_diffusion.diffusion_dataset import DiffusionWindowDataset
from carla_diffusion.diffusion_evaluation import (
    summarize_action_pairs,
    summarize_chunk_smoothness,
)
from carla_diffusion.diffusion_policy import DiffusionSchedule
from carla_diffusion.factorized_evaluation import factorized_promotion_decision
from carla_diffusion.temporal_bc_model import TemporalBC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factorized-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / "phase7_factorized_v190"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / "phase5_temporal_bc_v080"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase3_pilot_v1",
    )
    parser.add_argument(
        "--latency-report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "evaluations"
            / "phase7_factorized_latency_v190.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "evaluations"
            / "phase7_factorized_offline_gate_v190.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_temporal_bc(config: dict[str, Any]) -> TemporalBC:
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


def make_loader(
    dataset: DiffusionWindowDataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def evaluate_factorized(
    model: Any,
    schedule: DiffusionSchedule,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    noise_seed: int,
    inference_steps: int,
    eta: float,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    predicted_chunks: list[list[list[float]]] = []
    target_chunks: list[list[list[float]]] = []
    for host_batch in loader:
        batch = move_batch(host_batch, device)
        initial_noise = torch.randn(
            batch["target"].shape, generator=generator, device=device
        )
        prediction = model.sample(
            schedule,
            batch["images"],
            batch["state_history"],
            batch["condition"],
            inference_steps=inference_steps,
            eta=eta,
            initial_noise=initial_noise,
        )
        predicted_chunks.extend(prediction.float().cpu().tolist())
        target_chunks.extend(batch["target"].float().cpu().tolist())
    first_predictions = [chunk[0] for chunk in predicted_chunks]
    first_targets = [chunk[0] for chunk in target_chunks]
    flat_predictions = [action for chunk in predicted_chunks for action in chunk]
    flat_targets = [action for chunk in target_chunks for action in chunk]
    return {
        "noise_seed": noise_seed,
        "first_action": summarize_action_pairs(first_predictions, first_targets),
        "full_action_chunk": summarize_action_pairs(flat_predictions, flat_targets),
        "predicted_chunk_smoothness": summarize_chunk_smoothness(predicted_chunks),
        "target_chunk_smoothness": summarize_chunk_smoothness(target_chunks),
    }


@torch.inference_mode()
def evaluate_baseline(
    model: TemporalBC,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    predictions: list[list[float]] = []
    targets: list[list[float]] = []
    for host_batch in loader:
        batch = move_batch(host_batch, device)
        prediction = model(
            batch["images"], batch["state_history"], batch["condition"]
        )
        predictions.extend(prediction.float().cpu().tolist())
        targets.extend(batch["target"][:, 0].float().cpu().tolist())
    return summarize_action_pairs(predictions, targets)


def read_latency_gate(path: Path, checkpoint: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"passed": False, "reason": "latency report is missing"}
    report = json.loads(path.read_text(encoding="utf-8"))
    same_checkpoint = Path(str(report.get("checkpoint", ""))).resolve() == checkpoint
    return {
        "passed": bool(report.get("p95_latency_gate_passed")) and same_checkpoint,
        "same_checkpoint": same_checkpoint,
        "p95_ms": report.get("latency", {}).get("p95_ms"),
        "budget_ms": report.get("latency_budget_ms"),
        "report": str(path.resolve()),
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch size must be positive and workers cannot be negative")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required; pass --allow-cpu only for diagnostics")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_path = args.factorized_checkpoint.resolve()
    baseline_path = args.baseline_checkpoint.resolve()
    candidate_checkpoint: dict[str, Any] = torch.load(
        candidate_path, map_location=device, weights_only=False
    )
    baseline_checkpoint: dict[str, Any] = torch.load(
        baseline_path, map_location=device, weights_only=False
    )
    if candidate_checkpoint.get("model_type") != "factorized_temporal_policy":
        raise ValueError("candidate checkpoint has the wrong model type")
    project_config = candidate_checkpoint["config"]
    config = project_config["factorized_policy"]
    current_config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    thresholds = current_config["factorized_offline_evaluation"]
    model = build_model(config, pretrained=False).to(device)
    model.load_state_dict(candidate_checkpoint["model_state_dict"])
    model.eval()
    baseline_config = baseline_checkpoint["config"]["temporal_behavioral_cloning"]
    baseline = build_temporal_bc(baseline_config).to(device)
    baseline.load_state_dict(baseline_checkpoint["model_state_dict"])
    baseline.eval()
    schedule = DiffusionSchedule(
        int(config["diffusion_steps"]), float(config["cosine_s"])
    ).to(device)
    options = {
        "history_frames": int(config["history_frames"]),
        "action_horizon": int(config["action_horizon"]),
        "image_size": int(config["image_size"]),
        "normalized_state_clip": float(config["normalized_state_clip"]),
    }
    processed_root = args.processed_root.resolve()
    validation_dataset = DiffusionWindowDataset(
        processed_root, "validation", **options
    )
    test_dataset = DiffusionWindowDataset(processed_root, "test", **options)
    validation_loader = make_loader(
        validation_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
    )
    started = time.perf_counter()
    validation_candidates = [
        evaluate_factorized(
            model,
            schedule,
            validation_loader,
            device,
            noise_seed=int(seed),
            inference_steps=int(config["inference_steps"]),
            eta=float(config["ddim_eta"]),
        )
        for seed in thresholds["candidate_noise_seeds"]
    ]
    selected = min(
        validation_candidates,
        key=lambda candidate: candidate["first_action"]["joint_rmse"],
    )
    selected_seed = int(selected["noise_seed"])
    candidate_test = evaluate_factorized(
        model,
        schedule,
        test_loader,
        device,
        noise_seed=selected_seed,
        inference_steps=int(config["inference_steps"]),
        eta=float(config["ddim_eta"]),
    )
    baseline_test = evaluate_baseline(baseline, test_loader, device)
    latency = read_latency_gate(args.latency_report.resolve(), candidate_path)
    promotion = factorized_promotion_decision(
        candidate_test["first_action"],
        baseline_test,
        predicted_smoothness=candidate_test["predicted_chunk_smoothness"],
        target_smoothness=candidate_test["target_chunk_smoothness"],
        expected_validation_samples=len(validation_dataset),
        validation_samples=int(selected["first_action"]["samples"]),
        expected_test_samples=len(test_dataset),
        maximum_relative_rmse=float(thresholds["maximum_relative_rmse"]),
        maximum_absolute_longitudinal_bias=float(
            thresholds["maximum_absolute_longitudinal_bias"]
        ),
        maximum_longitudinal_behavior_drift=float(
            thresholds["maximum_longitudinal_behavior_drift"]
        ),
        maximum_longitudinal_smoothness_ratio=float(
            thresholds["maximum_longitudinal_smoothness_ratio"]
        ),
        latency_gate_passed=bool(latency["passed"]),
    )
    report = {
        "status": "passed",
        "model_type": "factorized_temporal_policy",
        "promotion_gate_passed": promotion["gate_passed"],
        "claim_boundary": (
            "Diffusion models steering only; longitudinal control is deterministic."
        ),
        "factorized_checkpoint": str(candidate_path),
        "factorized_checkpoint_sha256": file_sha256(candidate_path),
        "factorized_checkpoint_epoch": int(candidate_checkpoint["epoch"]),
        "baseline_checkpoint": str(baseline_path),
        "baseline_checkpoint_sha256": file_sha256(baseline_path),
        "device": str(device),
        "dataset_sizes": {
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
        "selection_protocol": "minimum validation first-action joint RMSE",
        "selected_noise_seed": selected_seed,
        "validation_candidates": validation_candidates,
        "factorized_test": candidate_test,
        "temporal_bc_test": baseline_test,
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
