#!/usr/bin/env python3
"""Calibrate and freeze a causal VLA steering smoother on validation data."""

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
from carla_diffusion.vla_dataset import VLAWindowDataset
from carla_diffusion.vla_smoothing import (
    SMOOTHER_TYPE,
    select_pareto_steering_smoothing,
    select_steering_smoothing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/checkpoints/phase8_vla_corrective_v240/best.pt"
        ),
    )
    parser.add_argument(
        "--base-training-report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/checkpoints/phase8_vla_corrective_v240/report.json"
        ),
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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/checkpoints/phase8_vla_smoothed_v250",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--selection-strategy",
        choices=("least_invasive", "pareto_robust"),
        default="least_invasive",
    )
    parser.add_argument(
        "--prior-gate-report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/evaluations/phase8_vla_smoothed_gate_v250.json"
        ),
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prior_test_exposure(
    path: Path,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Validate and disclose the already-observed Phase 8.3 test result."""
    report = json.loads(path.read_text(encoding="utf-8"))
    failures = report.get("promotion", {}).get("failures")
    if (
        report.get("status") != "passed"
        or report.get("promotion_gate_passed") is not False
        or report.get("checkpoint_sha256") != expected_checkpoint_sha256
        or failures != ["chunk_smoothness"]
    ):
        raise ValueError("prior Phase 8.3 gate report does not match frozen evidence")
    ratios = report["promotion"]["chunk_smoothness_ratio"]
    return {
        "split": "test",
        "report": str(path),
        "report_sha256": sha256(path),
        "checkpoint_sha256": str(report["checkpoint_sha256"]),
        "observed_failure": "chunk_smoothness",
        "observed_steering_smoothness_ratio": float(ratios["steering"]),
        "used_for_pareto_selection": False,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch size must be positive and workers cannot be negative")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required; pass --allow-cpu only for diagnostics")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_config = load_and_validate_config(PROJECT_ROOT / "configs/project.json")
    pareto_mode = args.selection_strategy == "pareto_robust"
    smoothing = project_config[
        "vla_pareto_smoothing" if pareto_mode else "vla_steering_smoothing"
    ]
    exposure = None
    if pareto_mode:
        exposure = prior_test_exposure(
            args.prior_gate_report.resolve(),
            str(smoothing["prior_smoothed_checkpoint_sha256"]),
        )
    base_path = args.base_checkpoint.resolve()
    base_digest = sha256(base_path)
    if base_digest != smoothing["base_checkpoint_sha256"]:
        raise ValueError("base checkpoint SHA-256 does not match the frozen Phase 8.3 input")
    base_report_path = args.base_training_report.resolve()
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    base_checkpoint: dict[str, Any] = torch.load(
        base_path, map_location=device, weights_only=False
    )
    if base_checkpoint.get("model_type") != "hierarchical_vla_preflight":
        raise ValueError("base checkpoint has the wrong model type")
    if (
        base_report.get("status") != "passed"
        or base_report.get("checkpoint_sha256") != base_digest
        or base_report.get("best_epoch") != base_checkpoint.get("epoch")
    ):
        raise ValueError("base checkpoint and training report provenance do not match")

    model_config = base_checkpoint["config"]["vision_language_action"]
    dataset = VLAWindowDataset(
        args.processed_root.resolve(),
        str(smoothing["calibration_split"]),
        history_frames=int(model_config["history_frames"]),
        planner_horizon=int(model_config["action_horizon"]),
        image_size=int(model_config["image_size"]),
        normalized_state_clip=float(model_config["normalized_state_clip"]),
        dataset_root=args.dataset_root.resolve(),
    )
    expected_samples = int(project_config["vla_offline_evaluation"]["expected_validation_samples"])
    if len(dataset) != expected_samples:
        raise ValueError(
            f"validation coverage mismatch: expected {expected_samples}, found {len(dataset)}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_model(base_checkpoint).to(device)
    model.load_state_dict(base_checkpoint["model_state_dict"])
    model.eval()
    predicted_chunks: list[list[list[float]]] = []
    target_chunks: list[list[list[float]]] = []
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

    calibration_kwargs = {
        "target_maximum_smoothness_ratio": float(
            smoothing["target_maximum_steering_smoothness_ratio"]
        ),
        "maximum_full_chunk_steering_rmse_degradation_fraction": float(
            smoothing["maximum_full_chunk_steering_rmse_degradation_fraction"]
        ),
    }
    if pareto_mode:
        calibration = select_pareto_steering_smoothing(
            predicted_chunks,
            target_chunks,
            smoothing["candidate_alphas"],
            **calibration_kwargs,
            rmse_equivalence_tolerance_fraction=float(
                smoothing["rmse_equivalence_tolerance_fraction"]
            ),
        )
    else:
        calibration = select_steering_smoothing(
            predicted_chunks,
            target_chunks,
            smoothing["candidate_alphas"],
            **calibration_kwargs,
        )
    if calibration["status"] != "passed":
        failed_report = {
            "status": "failed",
            "model_type": "hierarchical_vla_smoothed",
            "base_checkpoint": str(base_path),
            "base_checkpoint_sha256": base_digest,
            "validation_calibration": calibration,
            "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
        }
        atomic_json(output_dir / "report.json", failed_report)
        print(json.dumps(failed_report, indent=2))
        return 2

    transform = {
        "type": SMOOTHER_TYPE,
        "protocol_version": str(smoothing["protocol_version"]),
        "alpha": float(calibration["selected_alpha"]),
        "preserve_first_action": True,
        "steering_only": True,
        "causal": True,
        "calibrated_split": str(smoothing["calibration_split"]),
        "calibration_samples": len(dataset),
        "base_checkpoint_sha256": base_digest,
        "selection_rule": str(calibration["selection_rule"]),
        "target_maximum_steering_smoothness_ratio": float(
            smoothing["target_maximum_steering_smoothness_ratio"]
        ),
        "selection_strategy": str(args.selection_strategy),
        "rmse_equivalence_tolerance_fraction": (
            float(smoothing["rmse_equivalence_tolerance_fraction"])
            if pareto_mode
            else None
        ),
        "prior_test_exposure": exposure,
        "requires_fresh_holdout": bool(
            smoothing.get("requires_fresh_holdout", False)
        ),
    }
    derived_checkpoint = dict(base_checkpoint)
    derived_checkpoint["base_model_type"] = base_checkpoint["model_type"]
    derived_checkpoint["model_type"] = "hierarchical_vla_smoothed"
    derived_checkpoint["deployment_transform"] = transform
    derived_checkpoint[
        "phase8_4_config" if pareto_mode else "phase8_3_config"
    ] = dict(smoothing)
    checkpoint_path = output_dir / "best.pt"
    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(derived_checkpoint, temporary_checkpoint)
    os.replace(temporary_checkpoint, checkpoint_path)
    checkpoint_digest = sha256(checkpoint_path)
    report = {
        "status": "passed",
        "model_type": "hierarchical_vla_smoothed",
        "mode": (
            "validation_only_pareto_calibration"
            if pareto_mode
            else "validation_only_calibration"
        ),
        "output_dir": str(output_dir),
        "device": str(device),
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": base_digest,
        "base_training_report": str(base_report_path),
        "base_training_report_sha256": sha256(base_report_path),
        "best_epoch": int(base_checkpoint["epoch"]),
        "dataset_sizes": {
            "validation": len(dataset),
            "test": int(base_report["dataset_sizes"]["test"]),
        },
        "deployment_transform": transform,
        "validation_calibration": calibration,
        "prior_test_exposure": exposure,
        "fresh_holdout_required": bool(
            smoothing.get("requires_fresh_holdout", False)
        ),
        "checkpoint_sha256": checkpoint_digest,
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
        "claim_boundary": (
            "Validation-calibrated deterministic steering postprocessor. Existing test "
            "metrics were not used for candidate selection; however, that split was "
            "previously observed and cannot support an unbiased generalization or "
            "closed-loop claim."
            if pareto_mode
            else "Validation-calibrated deterministic steering postprocessor; test "
            "data was not read and no closed-loop claim is made."
        ),
    }
    atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
