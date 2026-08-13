#!/usr/bin/env python3
"""Train and evaluate the temporal action-chunk diffusion policy."""

from __future__ import annotations

import argparse
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
from carla_diffusion.diffusion_policy import (
    DiffusionSchedule,
    TemporalDiffusionPolicy,
    weighted_noise_mse,
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
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "phase7_diffusion",
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


def build_model(config: dict[str, Any], *, pretrained: bool) -> TemporalDiffusionPolicy:
    return TemporalDiffusionPolicy(
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
    generator = torch.Generator().manual_seed(seed)
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
    model: TemporalDiffusionPolicy,
    schedule: DiffusionSchedule,
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
        target = batch["target"]
        noise = torch.randn_like(target)
        timesteps = torch.randint(
            schedule.steps, (target.shape[0],), device=device, dtype=torch.long
        )
        noisy_actions = schedule.add_noise(target, noise, timesteps)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(
                noisy_actions,
                timesteps,
                batch["images"],
                batch["state_history"],
                batch["condition"],
            )
            loss = weighted_noise_mse(prediction, noise, batch["weight"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite diffusion loss at batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.detach())
        batches += 1
    if batches == 0:
        raise RuntimeError("diffusion training loader produced no batches")
    return loss_sum / batches


@torch.inference_mode()
def evaluate_noise(
    model: TemporalDiffusionPolicy,
    schedule: DiffusionSchedule,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    seed: int,
    max_batches: int | None,
) -> dict[str, float | int]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    squared_error = 0.0
    weighted_error = 0.0
    weight_sum = 0.0
    values = 0
    samples = 0
    for batch_index, host_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = to_device(host_batch, device)
        target = batch["target"]
        noise = torch.randn(target.shape, generator=generator, device=device)
        timesteps = torch.randint(
            schedule.steps,
            (target.shape[0],),
            generator=generator,
            device=device,
        )
        noisy_actions = schedule.add_noise(target, noise, timesteps)
        prediction = model(
            noisy_actions,
            timesteps,
            batch["images"],
            batch["state_history"],
            batch["condition"],
        )
        error = torch.square(prediction.float() - noise.float())
        per_sample = torch.mean(error, dim=(1, 2))
        weights = batch["weight"].reshape(-1)
        squared_error += float(torch.sum(error))
        weighted_error += float(torch.sum(per_sample * weights))
        weight_sum += float(torch.sum(weights))
        values += error.numel()
        samples += target.shape[0]
    if samples == 0:
        raise RuntimeError("diffusion evaluation loader produced no batches")
    return {
        "samples": samples,
        "noise_mse": squared_error / values,
        "weighted_noise_mse": weighted_error / max(weight_sum, 1e-8),
    }


@torch.inference_mode()
def evaluate_sampling(
    model: TemporalDiffusionPolicy,
    schedule: DiffusionSchedule,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    seed: int,
    inference_steps: int,
    max_batches: int,
) -> dict[str, float | int]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    absolute_error = torch.zeros(2, dtype=torch.float64)
    squared_error = torch.zeros(2, dtype=torch.float64)
    count = 0
    for batch_index, host_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = to_device(host_batch, device)
        initial_noise = torch.randn(
            batch["target"].shape,
            generator=generator,
            device=device,
        )
        prediction = schedule.ddim_sample(
            model,
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
        raise RuntimeError("diffusion sampling evaluation produced no batches")
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    project_config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    config = project_config["diffusion_policy"]
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
    sampling_eval_batches = int(config["sampling_evaluation_batches"])
    if args.smoke:
        epochs = 1
        max_train_batches = max_train_batches or 2
        max_eval_batches = max_eval_batches or 2
        sampling_eval_batches = 1

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_root = args.processed_root.resolve()
    dataset_options = {
        "history_frames": int(config["history_frames"]),
        "action_horizon": int(config["action_horizon"]),
        "image_size": int(config["image_size"]),
        "normalized_state_clip": float(config["normalized_state_clip"]),
    }
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
    model = build_model(
        config, pretrained=args.pretrained and not args.smoke
    ).to(device)
    schedule = DiffusionSchedule(
        steps=int(config["diffusion_steps"]), cosine_s=float(config["cosine_s"])
    ).to(device)
    freeze_epochs = 0 if args.smoke else int(config["freeze_encoder_epochs"])
    model.set_encoder_trainable(freeze_epochs == 0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_validation = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "best.pt"
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            model.set_encoder_trainable(True)
        training_loss = train_epoch(
            model,
            schedule,
            loaders["train"],
            optimizer,
            scaler,
            device,
            max_train_batches,
        )
        validation = evaluate_noise(
            model,
            schedule,
            loaders["validation"],
            device,
            seed=seed + epoch,
            max_batches=max_eval_batches,
        )
        record = {"epoch": epoch, "training_weighted_noise_mse": training_loss, **validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        metric = float(validation["weighted_noise_mse"])
        if not math.isfinite(metric):
            raise FloatingPointError(f"non-finite validation metric at epoch {epoch}")
        if metric < best_validation:
            best_validation = metric
            best_epoch = epoch
            epochs_without_improvement = 0
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model_type": "temporal_diffusion_policy",
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "config": project_config,
                    "pretrained": args.pretrained and not args.smoke,
                    "validation": validation,
                },
                temporary,
            )
            os.replace(temporary, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["early_stopping_patience"]):
                break
    if not checkpoint_path.is_file():
        raise RuntimeError("diffusion training completed without a finite checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_noise = evaluate_noise(
        model,
        schedule,
        loaders["test"],
        device,
        seed=seed + 1000,
        max_batches=max_eval_batches,
    )
    test_sampling = evaluate_sampling(
        model,
        schedule,
        loaders["test"],
        device,
        seed=seed + 2000,
        inference_steps=int(config["inference_steps"]),
        max_batches=sampling_eval_batches,
    )
    report = {
        "status": "passed",
        "model_type": "temporal_diffusion_policy",
        "mode": "smoke" if args.smoke else "full",
        "processed_root": str(processed_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "best_epoch": best_epoch,
        "validation_selection_metric": "weighted_noise_mse",
        "best_validation_weighted_noise_mse": best_validation,
        "test_noise_metrics": test_noise,
        "test_sampling_metrics": test_sampling,
        "training_hyperparameters": {
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "diffusion_steps": int(config["diffusion_steps"]),
            "inference_steps": int(config["inference_steps"]),
            "action_horizon": int(config["action_horizon"]),
            "execute_steps": int(config["execute_steps"]),
            "encoder_freeze_epochs": freeze_epochs,
        },
        "epochs": history,
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
