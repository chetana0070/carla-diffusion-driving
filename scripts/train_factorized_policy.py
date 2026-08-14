#!/usr/bin/env python3
"""Train steering diffusion with a deterministic longitudinal action-chunk head."""

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
from carla_diffusion.diffusion_policy import DiffusionSchedule
from carla_diffusion.factorized_policy import (
    FactorizedTemporalPolicy,
    load_diffusion_warm_start,
    weighted_longitudinal_derivative_loss,
    weighted_longitudinal_loss,
    weighted_mode_classification_loss,
)


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
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "phase7_factorized",
    )
    parser.add_argument(
        "--initial-diffusion-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / "phase7_diffusion_corrected_v180"
            / "best.pt"
        ),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--pretrained", action="store_true")
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


def build_model(config: dict[str, Any], *, pretrained: bool) -> FactorizedTemporalPolicy:
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
        pretrained=pretrained,
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


def weighted_axis_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("axis loss requires matching BxH tensors")
    per_sample = torch.mean(
        nn.functional.smooth_l1_loss(prediction, target, reduction="none"), dim=1
    )
    weights = sample_weights.reshape(-1)
    return torch.sum(per_sample * weights) / torch.clamp(torch.sum(weights), min=1e-8)


def weighted_axis_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("axis MSE requires matching BxH tensors")
    per_sample = torch.mean(torch.square(prediction - target), dim=1)
    weights = sample_weights.reshape(-1)
    return torch.sum(per_sample * weights) / torch.clamp(torch.sum(weights), min=1e-8)


def train_epoch(
    model: FactorizedTemporalPolicy,
    schedule: DiffusionSchedule,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    names = (
        "total",
        "steering_noise",
        "steering_action",
        "steering_derivative",
        "longitudinal_action",
        "longitudinal_derivative",
        "longitudinal_mode",
    )
    sums = dict.fromkeys(names, 0.0)
    mode_weights = torch.tensor(
        [
            config["longitudinal_mode_weights"]["braking"],
            config["longitudinal_mode_weights"]["neutral"],
            config["longitudinal_mode_weights"]["acceleration"],
        ],
        device=device,
        dtype=torch.float32,
    )
    batches = 0
    for batch_index, host_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(host_batch, device)
        target = batch["target"]
        noise = torch.randn_like(target)
        timesteps = torch.randint(
            schedule.steps, (target.shape[0],), device=device, dtype=torch.long
        )
        noisy = schedule.add_noise(target, noise, timesteps)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            predicted_noise, longitudinal, mode_logits = model(
                noisy,
                timesteps,
                batch["images"],
                batch["state_history"],
                batch["condition"],
            )
            predicted_clean = schedule.predict_clean_actions(
                noisy, predicted_noise, timesteps
            )
            steering_noise = weighted_axis_mse(
                predicted_noise[:, :, 0], noise[:, :, 0], batch["weight"]
            )
            steering_action = weighted_axis_loss(
                predicted_clean[:, :, 0], target[:, :, 0], batch["weight"]
            )
            steering_derivative = weighted_axis_loss(
                predicted_clean[:, 1:, 0] - predicted_clean[:, :-1, 0],
                target[:, 1:, 0] - target[:, :-1, 0],
                batch["weight"],
            )
            longitudinal_action = weighted_longitudinal_loss(
                longitudinal,
                target[:, :, 1],
                batch["weight"],
                mode_weights=mode_weights,
                neutral_threshold=float(config["longitudinal_neutral_threshold"]),
            )
            longitudinal_derivative = weighted_longitudinal_derivative_loss(
                longitudinal, target[:, :, 1], batch["weight"]
            )
            longitudinal_mode = weighted_mode_classification_loss(
                mode_logits,
                target[:, :, 1],
                batch["weight"],
                class_weights=mode_weights,
                neutral_threshold=float(config["longitudinal_neutral_threshold"]),
            )
            components = {
                "steering_noise": steering_noise,
                "steering_action": steering_action,
                "steering_derivative": steering_derivative,
                "longitudinal_action": longitudinal_action,
                "longitudinal_derivative": longitudinal_derivative,
                "longitudinal_mode": longitudinal_mode,
            }
            loss = sum(
                float(config[f"{name}_loss_weight"]) * value
                for name, value in components.items()
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite factorized loss at batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        sums["total"] += float(loss.detach())
        for name, value in components.items():
            sums[name] += float(value.detach())
        batches += 1
    if batches == 0:
        raise RuntimeError("factorized training loader produced no batches")
    return {name: value / batches for name, value in sums.items()}


@torch.inference_mode()
def evaluate_sampling(
    model: FactorizedTemporalPolicy,
    schedule: DiffusionSchedule,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    seed: int,
    inference_steps: int,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    absolute_error = torch.zeros(2, dtype=torch.float64)
    squared_error = torch.zeros(2, dtype=torch.float64)
    count = 0
    for batch_index, host_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
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
            initial_noise=initial_noise,
        )
        error = (prediction[:, 0].float() - batch["target"][:, 0]).cpu()
        absolute_error += torch.sum(torch.abs(error), dim=0).double()
        squared_error += torch.sum(torch.square(error), dim=0).double()
        count += error.shape[0]
    if count == 0:
        raise RuntimeError("factorized sampling evaluation produced no batches")
    mae = absolute_error / count
    rmse = torch.sqrt(squared_error / count)
    return {
        "samples": count,
        "first_action_steering_mae": float(mae[0]),
        "first_action_longitudinal_mae": float(mae[1]),
        "first_action_steering_rmse": float(rmse[0]),
        "first_action_longitudinal_rmse": float(rmse[1]),
        "first_action_joint_rmse": float(torch.sqrt(torch.mean(rmse**2))),
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    project_config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    config = project_config["factorized_policy"]
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
        raise RuntimeError(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_options = {
        "history_frames": int(config["history_frames"]),
        "action_horizon": int(config["action_horizon"]),
        "image_size": int(config["image_size"]),
        "normalized_state_clip": float(config["normalized_state_clip"]),
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
    model = build_model(config, pretrained=args.pretrained and not args.smoke).to(device)
    warm_start_path = args.initial_diffusion_checkpoint.resolve()
    warm_start_sha256: str | None = None
    if warm_start_path.is_file():
        checkpoint = torch.load(warm_start_path, map_location=device, weights_only=False)
        load_diffusion_warm_start(model, checkpoint)
        warm_start_sha256 = file_sha256(warm_start_path)
    elif not args.smoke:
        raise FileNotFoundError(f"released Phase 7.2 checkpoint is missing: {warm_start_path}")
    schedule = DiffusionSchedule(
        int(config["diffusion_steps"]), float(config["cosine_s"])
    ).to(device)
    freeze_epochs = 0 if args.smoke else int(config["freeze_encoder_epochs"])
    model.set_encoder_trainable(freeze_epochs == 0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    checkpoint_path = output_dir / "best.pt"
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            model.set_encoder_trainable(True)
        training = train_epoch(
            model,
            schedule,
            loaders["train"],
            optimizer,
            scaler,
            device,
            config,
            max_train_batches,
        )
        sampled = evaluate_sampling(
            model,
            schedule,
            loaders["validation"],
            device,
            seed=int(config["selection_noise_seed"]),
            inference_steps=int(config["inference_steps"]),
            max_batches=(1 if args.smoke else int(config["selection_sampling_batches"])),
        )
        record = {"epoch": epoch, "training_objective": training, "validation": sampled}
        history.append(record)
        print(json.dumps(record), flush=True)
        metric = float(sampled["first_action_joint_rmse"])
        if not math.isfinite(metric):
            raise FloatingPointError(f"non-finite validation metric at epoch {epoch}")
        if metric < best_validation:
            best_validation = metric
            best_epoch = epoch
            stale_epochs = 0
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model_type": "factorized_temporal_policy",
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "config": project_config,
                    "warm_start_checkpoint": str(warm_start_path),
                    "warm_start_checkpoint_sha256": warm_start_sha256,
                    "validation": sampled,
                },
                temporary,
            )
            os.replace(temporary, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["early_stopping_patience"]):
                break
    if not checkpoint_path.is_file():
        raise RuntimeError("factorized training produced no finite checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_sampling = evaluate_sampling(
        model,
        schedule,
        loaders["test"],
        device,
        seed=int(config["selection_noise_seed"]),
        inference_steps=int(config["inference_steps"]),
        max_batches=max_eval_batches,
    )
    report = {
        "status": "passed",
        "model_type": "factorized_temporal_policy",
        "mode": "smoke" if args.smoke else "full",
        "processed_root": str(processed_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "best_epoch": best_epoch,
        "validation_selection_metric": "sampled_first_action_joint_rmse",
        "best_validation_sampled_first_action_joint_rmse": best_validation,
        "warm_start_checkpoint": str(warm_start_path),
        "warm_start_checkpoint_sha256": warm_start_sha256,
        "test_sampling_metrics": test_sampling,
        "epochs": history,
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
