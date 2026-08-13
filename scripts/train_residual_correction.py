#!/usr/bin/env python3
"""Train a bounded recovery residual while keeping temporal BC fully frozen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.bc_model import weighted_action_mse
from carla_diffusion.residual_model import ResidualCorrection, corrected_action
from carla_diffusion.temporal_bc_dataset import TemporalWindowDataset
from carla_diffusion.temporal_bc_model import TemporalBC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase6_corrective_v2",
    )
    parser.add_argument(
        "--base-checkpoint",
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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "phase6_residual_v130",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def build_base(checkpoint: dict[str, Any], device: torch.device) -> TemporalBC:
    temporal = checkpoint["config"]["temporal_behavioral_cloning"]
    model = TemporalBC(
        history_frames=int(temporal["history_frames"]),
        state_dimension=int(temporal["state_input_dimension"]),
        condition_dimension=int(temporal["condition_input_dimension"]),
        image_projection_dimension=int(temporal["image_projection_dimension"]),
        state_projection_dimension=int(temporal["state_projection_dimension"]),
        temporal_hidden_dimension=int(temporal["temporal_hidden_dimension"]),
        condition_projection_dimension=int(temporal["condition_projection_dimension"]),
        temporal_layers=int(temporal["temporal_layers"]),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)
    model.eval()
    return model


def make_loader(
    dataset: TemporalWindowDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
    )


def device_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def prediction(
    base: TemporalBC,
    residual: ResidualCorrection,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        base_action = base(batch["images"], batch["state_history"], batch["condition"])
    delta = residual(batch["state_history"], batch["condition"], base_action)
    return base_action, corrected_action(base_action, delta)


@torch.inference_mode()
def evaluate(
    base: TemporalBC,
    residual: ResidualCorrection | None,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float | int]:
    base.eval()
    if residual is not None:
        residual.eval()
    absolute = torch.zeros(2, dtype=torch.float64)
    squared = torch.zeros(2, dtype=torch.float64)
    count = 0
    for host_batch in loader:
        batch = device_batch(host_batch, device)
        base_action = base(batch["images"], batch["state_history"], batch["condition"])
        output = base_action
        if residual is not None:
            output = corrected_action(
                base_action,
                residual(batch["state_history"], batch["condition"], base_action),
            )
        error = (output - batch["target"]).cpu()
        absolute += torch.sum(torch.abs(error), dim=0).double()
        squared += torch.sum(torch.square(error), dim=0).double()
        count += error.shape[0]
    if count == 0:
        raise RuntimeError("residual evaluation produced no samples")
    mae = absolute / count
    rmse = torch.sqrt(squared / count)
    return {
        "samples": count,
        "steering_mae": float(mae[0]),
        "longitudinal_mae": float(mae[1]),
        "steering_rmse": float(rmse[0]),
        "longitudinal_rmse": float(rmse[1]),
        "joint_rmse": float(torch.sqrt(torch.mean(rmse**2))),
    }


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for residual training")
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("invalid residual training hyperparameters")
    seed = 20260803
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    processed_root = args.processed_root.resolve()
    base_checkpoint_path = args.base_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
    base = build_base(checkpoint, device)
    temporal = checkpoint["config"]["temporal_behavioral_cloning"]
    options = {
        "history_frames": int(temporal["history_frames"]),
        "image_size": int(temporal["image_size"]),
        "normalized_state_clip": float(temporal["normalized_state_clip"]),
        "required_category": "corrective_intervention",
    }
    train_dataset = TemporalWindowDataset(processed_root, "train", augment=True, **options)
    validation_dataset = TemporalWindowDataset(
        processed_root, "correction_validation", **options
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        seed=seed,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=seed,
    )
    model_parameters = {
        "history_frames": int(temporal["history_frames"]),
        "state_dimension": int(temporal["state_input_dimension"]),
        "condition_dimension": int(temporal["condition_input_dimension"]),
        "hidden_dimensions": (128, 64),
        "maximum_steering_delta": 0.5,
        "maximum_longitudinal_delta": 0.5,
    }
    residual = ResidualCorrection(**model_parameters).to(device)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    initial_validation = evaluate(base, None, validation_loader, device)
    best_validation = float(initial_validation["joint_rmse"])
    best_epoch = 0
    patience = 5
    without_improvement = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "best.pt"
    started = time.perf_counter()

    def save_checkpoint(epoch: int, validation: dict[str, float | int]) -> None:
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model_type": "temporal_bc_gated_residual",
                "model_state_dict": residual.state_dict(),
                "epoch": epoch,
                "base_checkpoint": str(base_checkpoint_path),
                "base_checkpoint_sha256": hashlib.sha256(
                    base_checkpoint_path.read_bytes()
                ).hexdigest(),
                "config": checkpoint["config"],
                "model_parameters": model_parameters,
                "validation": validation,
            },
            temporary,
        )
        os.replace(temporary, checkpoint_path)

    save_checkpoint(0, initial_validation)
    for epoch in range(1, args.epochs + 1):
        residual.train()
        loss_sum = 0.0
        batches = 0
        for host_batch in train_loader:
            batch = device_batch(host_batch, device)
            optimizer.zero_grad(set_to_none=True)
            _, output = prediction(base, residual, batch)
            loss = weighted_action_mse(output, batch["target"], batch["weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1
        validation = evaluate(base, residual, validation_loader, device)
        record = {
            "epoch": epoch,
            "training_weighted_mse": loss_sum / max(1, batches),
            **validation,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        joint_rmse = float(validation["joint_rmse"])
        if not math.isfinite(joint_rmse):
            raise FloatingPointError("non-finite residual validation RMSE")
        if joint_rmse < best_validation:
            best_validation = joint_rmse
            best_epoch = epoch
            without_improvement = 0
            save_checkpoint(epoch, validation)
        else:
            without_improvement += 1
            if without_improvement >= patience:
                break
    report = {
        "status": "passed",
        "model_type": "temporal_bc_gated_residual",
        "base_checkpoint": str(base_checkpoint_path),
        "processed_root": str(processed_root),
        "dataset_sizes": {
            "correction_train": len(train_dataset),
            "correction_validation": len(validation_dataset),
        },
        "initial_corrective_validation": initial_validation,
        "best_epoch": best_epoch,
        "best_corrective_validation_joint_rmse": best_validation,
        "epochs": history,
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
