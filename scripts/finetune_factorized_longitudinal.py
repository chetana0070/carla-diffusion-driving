#!/usr/bin/env python3
"""Fine-tune only the deterministic longitudinal branch of Phase 7.3."""

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

from carla_diffusion.config import load_and_validate_config
from carla_diffusion.diffusion_dataset import DiffusionWindowDataset
from carla_diffusion.factorized_policy import (
    FactorizedTemporalPolicy,
    longitudinal_finetuning_loss,
    set_longitudinal_only_trainable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase3_pilot_v1",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / "phase7_factorized_v191"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / "phase7_factorized_longitudinal_v200"
        ),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--workers", type=int, default=2)
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(config: dict[str, Any]) -> FactorizedTemporalPolicy:
    return FactorizedTemporalPolicy(
        history_frames=int(config["history_frames"]),
        action_horizon=int(config["action_horizon"]),
        action_dimension=int(config["action_dimension"]),
        state_dimension=int(config["state_input_dimension"]),
        condition_dimension=int(config["condition_input_dimension"]),
        image_projection_dimension=int(config["image_projection_dimension"]),
        state_projection_dimension=int(config["state_projection_dimension"]),
        temporal_hidden_dimension=int(config["temporal_hidden_dimension"]),
        condition_projection_dimension=int(config["condition_projection_dimension"]),
        denoiser_dimension=int(config["denoiser_dimension"]),
        denoiser_layers=int(config["denoiser_layers"]),
        denoiser_heads=int(config["denoiser_heads"]),
        longitudinal_hidden_dimension=int(config["longitudinal_hidden_dimension"]),
        dropout=float(config["dropout"]),
    )


def make_loader(
    dataset: DiffusionWindowDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def longitudinal_prediction(
    model: FactorizedTemporalPolicy,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    with torch.no_grad():
        context = model.encode_observation(
            batch["images"], batch["state_history"], batch["condition"]
        )
    prediction, _ = model.predict_longitudinal(context)
    return prediction


def train_epoch(
    model: FactorizedTemporalPolicy,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    totals = {
        "objective": 0.0,
        "first_action_mse": 0.0,
        "chunk_mse": 0.0,
        "derivative_mse": 0.0,
    }
    batches = 0
    for batch_index, host_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(host_batch, device)
        target = batch["target"][:, :, 1]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = longitudinal_prediction(model, batch)
            loss, components = longitudinal_finetuning_loss(
                prediction,
                target,
                first_action_weight=float(config["first_action_mse_weight"]),
                chunk_weight=float(config["chunk_mse_weight"]),
                derivative_weight=float(config["derivative_mse_weight"]),
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"non-finite longitudinal loss at batch {batch_index}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=float(config["gradient_clip_norm"]),
        )
        scaler.step(optimizer)
        scaler.update()
        totals["objective"] += float(loss.detach())
        for name, value in components.items():
            totals[name] += float(value.detach())
        batches += 1
    if batches == 0:
        raise RuntimeError("longitudinal training loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


@torch.inference_mode()
def evaluate_longitudinal(
    model: FactorizedTemporalPolicy,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    max_batches: int | None,
    frozen_steering_rmse: float,
) -> dict[str, float | int]:
    model.eval()
    absolute_error = torch.zeros((), dtype=torch.float64)
    squared_error = torch.zeros((), dtype=torch.float64)
    bias = torch.zeros((), dtype=torch.float64)
    count = 0
    for batch_index, host_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(host_batch, device)
        prediction = longitudinal_prediction(model, batch)[:, 0].float()
        target = batch["target"][:, 0, 1].float()
        error = (prediction - target).cpu().double()
        absolute_error += torch.sum(torch.abs(error))
        squared_error += torch.sum(torch.square(error))
        bias += torch.sum(error)
        count += error.numel()
    if count == 0:
        raise RuntimeError("longitudinal evaluation produced no batches")
    rmse = math.sqrt(float(squared_error) / count)
    return {
        "samples": count,
        "first_action_longitudinal_mae": float(absolute_error) / count,
        "first_action_longitudinal_rmse": rmse,
        "first_action_longitudinal_bias": float(bias) / count,
        "frozen_steering_rmse": frozen_steering_rmse,
        "joint_rmse_proxy": math.sqrt((frozen_steering_rmse**2 + rmse**2) / 2),
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    project_config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    model_config = project_config["factorized_policy"]
    config = project_config["factorized_longitudinal_finetuning"]
    seed = int(project_config["project"]["random_seed"])
    seed_everything(seed)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required; pass --allow-cpu only for diagnostics")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = int(args.epochs or config["epochs"])
    batch_size = int(args.batch_size or config["batch_size"])
    learning_rate = float(args.learning_rate or config["learning_rate"])
    max_train_batches = args.max_train_batches
    max_eval_batches = args.max_eval_batches
    if args.smoke:
        epochs = 1
        max_train_batches = max_train_batches or 2
        max_eval_batches = max_eval_batches or 1

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"output directory is not empty: {output_dir}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.initial_checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Phase 7.3 checkpoint is missing: {checkpoint_path}")
    initial = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if initial.get("model_type") != "factorized_temporal_policy":
        raise ValueError("initial checkpoint is not a factorized temporal policy")
    model = build_model(model_config).to(device)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    set_longitudinal_only_trainable(model)
    frozen_steering_rmse = float(
        initial.get("validation", {}).get("first_action_steering_rmse", 0.0)
    )
    if frozen_steering_rmse <= 0:
        raise ValueError("initial checkpoint lacks validation steering RMSE")

    dataset_options = {
        "history_frames": int(model_config["history_frames"]),
        "action_horizon": int(model_config["action_horizon"]),
        "image_size": int(model_config["image_size"]),
        "normalized_state_clip": float(model_config["normalized_state_clip"]),
    }
    processed_root = args.processed_root.resolve()
    datasets = {
        "train": DiffusionWindowDataset(
            processed_root, "train", augment=True, **dataset_options
        ),
        "validation": DiffusionWindowDataset(
            processed_root, "validation", **dataset_options
        ),
        "test": DiffusionWindowDataset(processed_root, "test", **dataset_options),
    }
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
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_metric = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    best_path = output_dir / "best.pt"
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        training = train_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            config,
            max_train_batches,
        )
        validation = evaluate_longitudinal(
            model,
            loaders["validation"],
            device,
            max_batches=max_eval_batches,
            frozen_steering_rmse=frozen_steering_rmse,
        )
        record = {"epoch": epoch, "training": training, "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        metric = float(validation["first_action_longitudinal_rmse"])
        if not math.isfinite(metric):
            raise FloatingPointError(f"non-finite validation metric at epoch {epoch}")
        if metric < best_metric - float(config["minimum_improvement"]):
            best_metric = metric
            best_epoch = epoch
            stale_epochs = 0
            temporary = best_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model_type": "factorized_temporal_policy",
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "config": project_config,
                    "initial_checkpoint": str(checkpoint_path),
                    "initial_checkpoint_sha256": file_sha256(checkpoint_path),
                    "fine_tuning_scope": "longitudinal_only",
                    "validation": validation,
                },
                temporary,
            )
            os.replace(temporary, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["early_stopping_patience"]):
                break

    if not best_path.is_file():
        raise RuntimeError("longitudinal fine-tuning produced no finite checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    test = evaluate_longitudinal(
        model,
        loaders["test"],
        device,
        max_batches=max_eval_batches,
        frozen_steering_rmse=frozen_steering_rmse,
    )
    report = {
        "status": "passed",
        "model_type": "factorized_temporal_policy",
        "fine_tuning_scope": "longitudinal_only",
        "mode": "smoke" if args.smoke else "full",
        "processed_root": str(processed_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "initial_checkpoint": str(checkpoint_path),
        "initial_checkpoint_sha256": file_sha256(checkpoint_path),
        "best_epoch": best_epoch,
        "selection_metric": "validation_first_action_longitudinal_rmse",
        "best_validation_first_action_longitudinal_rmse": best_metric,
        "test_metrics": test,
        "epochs": history,
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
