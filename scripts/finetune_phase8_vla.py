#!/usr/bin/env python3
"""Correct Phase 8 VLA steering smoothness and longitudinal bias."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch import nn
from torch.utils.data import DataLoader

from carla_diffusion.config import load_and_validate_config
from carla_diffusion.diffusion_evaluation import (
    summarize_action_pairs,
    summarize_chunk_smoothness,
)
from carla_diffusion.vla_dataset import VLAWindowDataset
from carla_diffusion.vla_policy import (
    HierarchicalVLAPlanner,
    corrective_action_chunk_loss,
    corrective_selection_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase8_vla_v1",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Override the raw image root when processed data is relocated.",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "artifacts/checkpoints/phase8_vla_v220/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/checkpoints/phase8_vla_corrective_v240",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


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


def build_model(checkpoint: dict[str, Any]) -> HierarchicalVLAPlanner:
    config = checkpoint["config"]["vision_language_action"]
    return HierarchicalVLAPlanner(
        vocabulary_size=len(checkpoint["vocabulary"]),
        state_dimension=int(config["state_input_dimension"]),
        action_horizon=int(config["action_horizon"]),
        language_dimension=int(config["language_dimension"]),
        state_hidden_dimension=int(config["state_hidden_dimension"]),
        fusion_dimension=int(config["fusion_dimension"]),
    )


def make_loader(
    dataset: VLAWindowDataset,
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


def predict(
    model: HierarchicalVLAPlanner, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    return model(
        batch["images"],
        batch["state_history"],
        batch["instruction_token_ids"],
        batch["instruction_attention_mask"],
    )


def freeze_visual_encoder(model: HierarchicalVLAPlanner) -> None:
    for parameter in model.visual_encoder.parameters():
        parameter.requires_grad = False
    model.visual_encoder.eval()


def train_epoch(
    model: HierarchicalVLAPlanner,
    loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
    *,
    freeze_encoder: bool,
    max_batches: int | None,
) -> dict[str, float]:
    model.train()
    if freeze_encoder:
        model.visual_encoder.eval()
    totals: dict[str, float] = {"objective": 0.0}
    batches = 0
    for batch_index, host_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(host_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = predict(model, batch)
            loss, components = corrective_action_chunk_loss(
                prediction,
                batch["target"],
                batch["weight"],
                batch["condition_weight"],
                chunk_steering_weight=float(config["chunk_steering_mse_weight"]),
                chunk_longitudinal_weight=float(
                    config["chunk_longitudinal_mse_weight"]
                ),
                first_steering_weight=float(
                    config["first_action_steering_mse_weight"]
                ),
                first_longitudinal_weight=float(
                    config["first_action_longitudinal_mse_weight"]
                ),
                steering_derivative_weight=float(
                    config["steering_derivative_mse_weight"]
                ),
                longitudinal_derivative_weight=float(
                    config["longitudinal_derivative_mse_weight"]
                ),
                longitudinal_bias_weight=float(
                    config["longitudinal_bias_loss_weight"]
                ),
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite corrective loss at batch {batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(config["gradient_clip_norm"]),
        )
        scaler.step(optimizer)
        scaler.update()
        totals["objective"] += float(loss.detach())
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
        batches += 1
    if batches == 0:
        raise RuntimeError("corrective training loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: HierarchicalVLAPlanner,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    config: dict[str, Any],
    *,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    predicted_chunks: list[list[list[float]]] = []
    target_chunks: list[list[list[float]]] = []
    for batch_index, host_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(host_batch, device)
        predicted_chunks.extend(predict(model, batch).float().cpu().tolist())
        target_chunks.extend(batch["target"].float().cpu().tolist())
    if not predicted_chunks:
        raise RuntimeError("corrective evaluation produced no batches")
    first_action = summarize_action_pairs(
        [chunk[0] for chunk in predicted_chunks],
        [chunk[0] for chunk in target_chunks],
    )
    predicted_smoothness = summarize_chunk_smoothness(predicted_chunks)
    target_smoothness = summarize_chunk_smoothness(target_chunks)
    steering_ratio = predicted_smoothness["mean_abs_steering_step"] / max(
        target_smoothness["mean_abs_steering_step"], 1e-12
    )
    score, score_components = corrective_selection_score(
        joint_rmse=float(first_action["joint_rmse"]),
        longitudinal_bias=float(first_action["longitudinal"]["bias"]),
        steering_smoothness_ratio=steering_ratio,
        target_maximum_absolute_bias=float(
            config["target_maximum_absolute_longitudinal_bias"]
        ),
        target_maximum_smoothness_ratio=float(
            config["target_maximum_steering_smoothness_ratio"]
        ),
        bias_violation_weight=float(config["selection_bias_violation_weight"]),
        smoothness_violation_weight=float(
            config["selection_smoothness_violation_weight"]
        ),
    )
    return {
        "samples": int(first_action["samples"]),
        "first_action": first_action,
        "predicted_chunk_smoothness": predicted_smoothness,
        "target_chunk_smoothness": target_smoothness,
        "steering_smoothness_ratio": steering_ratio,
        "selection_score": score,
        "selection_components": score_components,
    }


def main() -> int:
    args = parse_args()
    project_config = load_and_validate_config(PROJECT_ROOT / "configs/project.json")
    vla = project_config["vision_language_action"]
    config = project_config["vla_corrective_finetuning"]
    seed = int(project_config["project"]["random_seed"])
    seed_everything(seed)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required; pass --allow-cpu only for diagnostics")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = int(args.epochs or config["epochs"])
    batch_size = int(args.batch_size or config["batch_size"])
    workers = int(
        args.workers if args.workers is not None else config["data_loader_workers"]
    )
    learning_rate = float(args.learning_rate or config["learning_rate"])
    if epochs < 1 or batch_size < 1 or workers < 0 or not 0 < learning_rate < 1:
        raise ValueError("invalid corrective fine-tuning runtime options")
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
    initial_path = args.initial_checkpoint.resolve()
    if not initial_path.is_file():
        raise FileNotFoundError(f"Phase 8.0 checkpoint is missing: {initial_path}")
    initial_sha256 = sha256(initial_path)
    if initial_sha256 != config["initial_checkpoint_sha256"]:
        raise ValueError("initial Phase 8.0 checkpoint digest does not match the release")
    initial: dict[str, Any] = torch.load(
        initial_path, map_location=device, weights_only=False
    )
    if initial.get("model_type") != "hierarchical_vla_preflight":
        raise ValueError("initial checkpoint is not a hierarchical VLA planner")
    processed_root = args.processed_root.resolve()
    dataset_vocabulary = json.loads(
        (processed_root / "vocabulary.json").read_text(encoding="utf-8")
    )
    if initial.get("vocabulary") != dataset_vocabulary:
        raise ValueError("initial checkpoint vocabulary does not match the dataset")
    model = build_model(initial).to(device)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    freeze_encoder = bool(config["freeze_visual_encoder"])
    if freeze_encoder:
        freeze_visual_encoder(model)

    dataset_options = {
        "history_frames": int(vla["history_frames"]),
        "planner_horizon": int(vla["action_horizon"]),
        "image_size": int(vla["image_size"]),
        "normalized_state_clip": float(vla["normalized_state_clip"]),
        "dataset_root": args.dataset_root,
    }
    datasets = {
        "train": VLAWindowDataset(
            processed_root,
            "train",
            augment=True,
            route_command_weights=config["route_command_weights"],
            traffic_light_weights=config["traffic_light_weights"],
            **dataset_options,
        ),
        "validation": VLAWindowDataset(
            processed_root, "validation", **dataset_options
        ),
        "test": VLAWindowDataset(processed_root, "test", **dataset_options),
    }
    loaders = {
        name: make_loader(
            dataset,
            batch_size=batch_size,
            workers=workers,
            shuffle=name == "train",
            seed=seed,
            device=device,
        )
        for name, dataset in datasets.items()
    }
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_score = float("inf")
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
            freeze_encoder=freeze_encoder,
            max_batches=max_train_batches,
        )
        validation = evaluate(
            model,
            loaders["validation"],
            device,
            config,
            max_batches=max_eval_batches,
        )
        record = {"epoch": epoch, "training": training, "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        score = float(validation["selection_score"])
        if not math.isfinite(score):
            raise FloatingPointError(f"non-finite selection score at epoch {epoch}")
        if score < best_score - float(config["minimum_improvement"]):
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            temporary = best_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "model_type": "hierarchical_vla_preflight",
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "config": project_config,
                    "vocabulary": initial["vocabulary"],
                    "initial_checkpoint": str(initial_path),
                    "initial_checkpoint_sha256": initial_sha256,
                    "fine_tuning_scope": "smoothness_and_longitudinal_bias",
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
        raise RuntimeError("corrective fine-tuning produced no finite checkpoint")
    best: dict[str, Any] = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    test = evaluate(
        model,
        loaders["test"],
        device,
        config,
        max_batches=max_eval_batches,
    )
    report = {
        "status": "passed",
        "model_type": "hierarchical_vla_preflight",
        "fine_tuning_scope": "smoothness_and_longitudinal_bias",
        "mode": "smoke" if args.smoke else "full",
        "processed_root": str(processed_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": initial_sha256,
        "visual_encoder_frozen": freeze_encoder,
        "best_epoch": best_epoch,
        "selection_metric": "validation_corrective_score",
        "best_validation_selection_score": best_score,
        "best_validation_metrics": best["validation"],
        "test_metrics": test,
        "checkpoint_sha256": sha256(best_path),
        "epochs": history,
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
        "claim_boundary": "Corrective offline candidate; no closed-loop claim.",
    }
    atomic_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
