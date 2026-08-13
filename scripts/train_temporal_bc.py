#!/usr/bin/env python3
"""Train and evaluate the four-frame temporal behavioral-cloning baseline."""

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
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.bc_model import weighted_action_mse
from carla_diffusion.config import load_and_validate_config
from carla_diffusion.temporal_bc_dataset import TemporalWindowDataset
from carla_diffusion.temporal_bc_model import TemporalBC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase3_pilot_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "phase5_temporal_bc",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--freeze-encoder-epochs", type=int)
    parser.add_argument(
        "--corrective-validation-split",
        choices=("correction_validation",),
    )
    parser.add_argument(
        "--max-nominal-validation-degradation-fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(
    dataset: TemporalWindowDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader[dict[str, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_epoch(
    model: TemporalBC,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.train()
    loss_sum = 0.0
    batches = 0
    for batch_index, host_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = to_device(host_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(
                batch["images"], batch["state_history"], batch["condition"]
            )
            loss = weighted_action_mse(prediction, batch["target"], batch["weight"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite temporal training loss at batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach())
        batches += 1
    if batches == 0:
        raise RuntimeError("temporal training loader produced no batches")
    return loss_sum / batches


@torch.inference_mode()
def evaluate(
    model: TemporalBC,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    steering_tolerance: float,
    longitudinal_tolerance: float,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    absolute_error = torch.zeros(2, dtype=torch.float64)
    squared_error = torch.zeros(2, dtype=torch.float64)
    tolerance_count = torch.zeros(2, dtype=torch.int64)
    joint_tolerance_count = 0
    count = 0
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    for batch_index, host_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = to_device(host_batch, device)
        prediction = model(batch["images"], batch["state_history"], batch["condition"])
        if not bool(torch.all(torch.isfinite(prediction))):
            raise FloatingPointError(
                f"non-finite temporal prediction at evaluation batch {batch_index}"
            )
        error = (prediction.float() - batch["target"]).cpu()
        batch_weights = batch["weight"].cpu().reshape(-1)
        per_sample_mse = torch.mean(torch.square(error), dim=1)
        weighted_loss_sum += float(torch.sum(per_sample_mse * batch_weights))
        weight_sum += float(torch.sum(batch_weights))
        absolute_error += torch.sum(torch.abs(error), dim=0).double()
        squared_error += torch.sum(torch.square(error), dim=0).double()
        within = torch.stack(
            (
                torch.abs(error[:, 0]) <= steering_tolerance,
                torch.abs(error[:, 1]) <= longitudinal_tolerance,
            ),
            dim=1,
        )
        tolerance_count += torch.sum(within, dim=0)
        joint_tolerance_count += int(torch.sum(torch.all(within, dim=1)))
        count += error.shape[0]
    if count == 0:
        raise RuntimeError("temporal evaluation loader produced no batches")
    mae = absolute_error / count
    rmse = torch.sqrt(squared_error / count)
    tolerance = tolerance_count.double() / count
    return {
        "samples": count,
        "weighted_mse": weighted_loss_sum / max(weight_sum, 1e-8),
        "steering_mae": float(mae[0]),
        "longitudinal_mae": float(mae[1]),
        "steering_rmse": float(rmse[0]),
        "longitudinal_rmse": float(rmse[1]),
        "joint_rmse": float(torch.sqrt(torch.mean(rmse**2))),
        "steering_tolerance_rate": float(tolerance[0]),
        "longitudinal_tolerance_rate": float(tolerance[1]),
        "joint_tolerance_rate": joint_tolerance_count / count,
    }


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_model(config: dict[str, Any], *, pretrained: bool) -> TemporalBC:
    return TemporalBC(
        history_frames=int(config["history_frames"]),
        state_dimension=int(config["state_input_dimension"]),
        condition_dimension=int(config["condition_input_dimension"]),
        image_projection_dimension=int(config["image_projection_dimension"]),
        state_projection_dimension=int(config["state_projection_dimension"]),
        temporal_hidden_dimension=int(config["temporal_hidden_dimension"]),
        condition_projection_dimension=int(config["condition_projection_dimension"]),
        temporal_layers=int(config["temporal_layers"]),
        pretrained=pretrained,
    )


def main() -> int:
    args = parse_args()
    config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    temporal_config = config["temporal_behavioral_cloning"]
    if args.pretrained and args.initial_checkpoint is not None:
        raise ValueError("--pretrained and --initial-checkpoint are mutually exclusive")
    if (
        args.corrective_validation_split is not None
        and args.initial_checkpoint is None
    ):
        raise ValueError("corrective checkpoint selection requires --initial-checkpoint")
    if args.max_nominal_validation_degradation_fraction < 0:
        raise ValueError("nominal validation degradation fraction cannot be negative")
    if args.freeze_encoder_epochs is not None and args.freeze_encoder_epochs < 0:
        raise ValueError("freeze encoder epochs cannot be negative")
    seed = int(config["project"]["random_seed"])
    seed_everything(seed)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required; pass --allow-cpu only for diagnostics")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = int(args.epochs or temporal_config["epochs"])
    batch_size = int(args.batch_size or temporal_config["batch_size"])
    learning_rate = float(args.learning_rate or temporal_config["learning_rate"])
    max_train_batches = args.max_train_batches
    max_eval_batches = args.max_eval_batches
    if args.smoke:
        epochs = 1
        max_train_batches = max_train_batches or 2
        max_eval_batches = max_eval_batches or 2

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_root = args.processed_root.resolve()
    dataset_options = {
        "history_frames": int(temporal_config["history_frames"]),
        "image_size": int(temporal_config["image_size"]),
        "normalized_state_clip": float(temporal_config["normalized_state_clip"]),
    }
    datasets = {
        "train": TemporalWindowDataset(
            processed_root, "train", augment=True, **dataset_options
        ),
        "validation": TemporalWindowDataset(processed_root, "validation", **dataset_options),
        "test": TemporalWindowDataset(processed_root, "test", **dataset_options),
    }
    if args.corrective_validation_split is not None:
        datasets["correction_validation"] = TemporalWindowDataset(
            processed_root,
            args.corrective_validation_split,
            **dataset_options,
        )
    loaders = {
        name: make_loader(
            dataset,
            batch_size=batch_size,
            workers=args.workers,
            shuffle=name == "train",
            seed=seed,
            device=device,
        )
        for name, dataset in datasets.items()
    }
    model = build_model(
        temporal_config,
        pretrained=args.pretrained and not args.smoke and args.initial_checkpoint is None,
    ).to(device)
    initial_checkpoint_path: Path | None = None
    initial_checkpoint_sha256: str | None = None
    initial_checkpoint_epoch: int | None = None
    if args.initial_checkpoint is not None:
        initial_checkpoint_path = args.initial_checkpoint.resolve()
        initial_checkpoint = torch.load(
            initial_checkpoint_path, map_location=device, weights_only=False
        )
        if initial_checkpoint.get("model_type") != "temporal_bc":
            raise ValueError("initial checkpoint is not a temporal BC checkpoint")
        model.load_state_dict(initial_checkpoint["model_state_dict"])
        initial_checkpoint_epoch = int(initial_checkpoint["epoch"])
        initial_checkpoint_sha256 = hashlib.sha256(
            initial_checkpoint_path.read_bytes()
        ).hexdigest()
    if args.freeze_encoder_epochs is not None:
        freeze_epochs = args.freeze_encoder_epochs
    elif args.smoke or initial_checkpoint_path is not None:
        freeze_epochs = 0
    else:
        freeze_epochs = int(temporal_config["freeze_encoder_epochs"])
    model.set_encoder_trainable(freeze_epochs == 0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(temporal_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_validation = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "best.pt"
    started = time.perf_counter()
    initial_validation: dict[str, Any] | None = None
    initial_corrective_validation: dict[str, Any] | None = None
    current_corrective_validation: dict[str, Any] | None = None
    best_corrective_validation = float("inf")

    def save_checkpoint(epoch: int, validation: dict[str, Any]) -> None:
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model_type": "temporal_bc",
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "config": config,
                "pretrained": args.pretrained and not args.smoke,
                "initial_checkpoint": (
                    str(initial_checkpoint_path) if initial_checkpoint_path else None
                ),
                "initial_checkpoint_sha256": initial_checkpoint_sha256,
                "initial_checkpoint_epoch": initial_checkpoint_epoch,
                "validation": validation,
                "corrective_validation": current_corrective_validation,
            },
            temporary,
        )
        os.replace(temporary, checkpoint_path)

    if initial_checkpoint_path is not None:
        initial_validation = evaluate(
            model,
            loaders["validation"],
            device,
            steering_tolerance=float(temporal_config["steering_tolerance"]),
            longitudinal_tolerance=float(temporal_config["longitudinal_tolerance"]),
            max_batches=max_eval_batches,
        )
        best_validation = float(initial_validation["joint_rmse"])
        if not math.isfinite(best_validation):
            raise FloatingPointError("non-finite initial-checkpoint validation RMSE")
        if "correction_validation" in loaders:
            initial_corrective_validation = evaluate(
                model,
                loaders["correction_validation"],
                device,
                steering_tolerance=float(temporal_config["steering_tolerance"]),
                longitudinal_tolerance=float(temporal_config["longitudinal_tolerance"]),
                max_batches=max_eval_batches,
            )
            best_corrective_validation = float(
                initial_corrective_validation["joint_rmse"]
            )
        current_corrective_validation = initial_corrective_validation
        save_checkpoint(0, initial_validation)

    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            model.set_encoder_trainable(True)
        training_loss = train_epoch(
            model, loaders["train"], optimizer, scaler, device, max_train_batches
        )
        validation = evaluate(
            model,
            loaders["validation"],
            device,
            steering_tolerance=float(temporal_config["steering_tolerance"]),
            longitudinal_tolerance=float(temporal_config["longitudinal_tolerance"]),
            max_batches=max_eval_batches,
        )
        corrective_validation = None
        if "correction_validation" in loaders:
            corrective_validation = evaluate(
                model,
                loaders["correction_validation"],
                device,
                steering_tolerance=float(temporal_config["steering_tolerance"]),
                longitudinal_tolerance=float(temporal_config["longitudinal_tolerance"]),
                max_batches=max_eval_batches,
            )
        record = {
            "epoch": epoch,
            "training_weighted_mse": training_loss,
            **validation,
            "corrective_validation": corrective_validation,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        joint_rmse = float(validation["joint_rmse"])
        if not math.isfinite(joint_rmse):
            raise FloatingPointError(f"non-finite temporal validation RMSE at epoch {epoch}")
        if corrective_validation is None:
            improved = joint_rmse < best_validation
        else:
            if initial_validation is None:
                raise RuntimeError("corrective selection requires an initial checkpoint")
            nominal_limit = float(initial_validation["joint_rmse"]) * (
                1.0 + args.max_nominal_validation_degradation_fraction
            )
            corrective_rmse = float(corrective_validation["joint_rmse"])
            improved = (
                joint_rmse <= nominal_limit
                and corrective_rmse < best_corrective_validation
            )
        if improved:
            best_validation = joint_rmse
            if corrective_validation is not None:
                best_corrective_validation = float(
                    corrective_validation["joint_rmse"]
                )
            best_epoch = epoch
            epochs_without_improvement = 0
            current_corrective_validation = corrective_validation
            save_checkpoint(epoch, validation)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(
                temporal_config["early_stopping_patience"]
            ):
                break

    if not checkpoint_path.is_file():
        raise RuntimeError("temporal training completed without a finite checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model,
        loaders["test"],
        device,
        steering_tolerance=float(temporal_config["steering_tolerance"]),
        longitudinal_tolerance=float(temporal_config["longitudinal_tolerance"]),
        max_batches=max_eval_batches,
    )
    report = {
        "status": "passed",
        "model_type": "temporal_bc",
        "mode": "smoke" if args.smoke else "full",
        "processed_root": str(processed_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "pretrained": args.pretrained and not args.smoke,
        "initial_checkpoint": (
            str(initial_checkpoint_path) if initial_checkpoint_path else None
        ),
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "initial_checkpoint_epoch": initial_checkpoint_epoch,
        "initial_validation_metrics": initial_validation,
        "initial_corrective_validation_metrics": initial_corrective_validation,
        "training_hyperparameters": {
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": float(temporal_config["weight_decay"]),
            "encoder_freeze_epochs": freeze_epochs,
            "corrective_validation_split": args.corrective_validation_split,
            "max_nominal_validation_degradation_fraction": (
                args.max_nominal_validation_degradation_fraction
            ),
        },
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "best_epoch": best_epoch,
        "validation_selection_metric": "joint_rmse",
        "best_validation_joint_rmse": best_validation,
        "best_corrective_validation_joint_rmse": (
            best_corrective_validation
            if math.isfinite(best_corrective_validation)
            else None
        ),
        "test_metrics": test_metrics,
        "epochs": history,
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
