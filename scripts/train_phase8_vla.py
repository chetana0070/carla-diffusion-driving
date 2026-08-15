#!/usr/bin/env python3
"""Train the compact hierarchical VLA planner on frozen Phase 8 windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from carla_diffusion.config import load_and_validate_config
from carla_diffusion.vla_dataset import VLAWindowDataset
from carla_diffusion.vla_policy import HierarchicalVLAPlanner, weighted_action_chunk_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase8_vla_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "checkpoints" / "phase8_vla_preflight_v220",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Override the raw image root when processed data is relocated, such as on SOL.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--pretrained-visual-encoder", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def limited(dataset: Dataset[dict[str, torch.Tensor]], smoke: bool) -> Dataset[Any]:
    if not smoke:
        return dataset
    return Subset(dataset, range(min(16, len(dataset))))


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


def evaluate(
    model: HierarchicalVLAPlanner,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    samples = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = move(raw_batch, device)
            prediction = model(
                batch["images"],
                batch["state_history"],
                batch["instruction_token_ids"],
                batch["instruction_attention_mask"],
            )
            difference = prediction - batch["target"]
            squared_error += float(torch.sum(difference**2))
            absolute_error += float(torch.sum(torch.abs(difference)))
            samples += int(prediction.shape[0])
    elements = samples * model.action_horizon * 2
    return {
        "samples": samples,
        "action_chunk_rmse": (squared_error / elements) ** 0.5,
        "action_chunk_mae": absolute_error / elements,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    vla = config["vision_language_action"]
    seed = int(config["project"]["random_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(seed)
    elif args.allow_cpu or args.smoke:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is required for full VLA training; use --allow-cpu deliberately")

    dataset_kwargs = {
        "history_frames": int(vla["history_frames"]),
        "planner_horizon": int(vla["action_horizon"]),
        "image_size": int(vla["image_size"]),
        "normalized_state_clip": float(vla["normalized_state_clip"]),
        "dataset_root": args.dataset_root,
    }
    processed_root = args.processed_root.resolve()
    train_dataset = limited(
        VLAWindowDataset(processed_root, "train", augment=True, **dataset_kwargs),
        args.smoke,
    )
    validation_dataset = limited(
        VLAWindowDataset(processed_root, "validation", **dataset_kwargs),
        args.smoke,
    )
    test_dataset = limited(
        VLAWindowDataset(processed_root, "test", **dataset_kwargs),
        args.smoke,
    )
    vocabulary: dict[str, int] = json.loads(
        (processed_root / "vocabulary.json").read_text(encoding="utf-8")
    )
    model = HierarchicalVLAPlanner(
        vocabulary_size=len(vocabulary),
        state_dimension=int(vla["state_input_dimension"]),
        action_horizon=int(vla["action_horizon"]),
        language_dimension=int(vla["language_dimension"]),
        state_hidden_dimension=int(vla["state_hidden_dimension"]),
        fusion_dimension=int(vla["fusion_dimension"]),
        pretrained_visual_encoder=args.pretrained_visual_encoder,
    ).to(device)
    batch_size = min(int(vla["batch_size"]), len(train_dataset))
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": int(vla["data_loader_workers"]),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(vla["learning_rate"]),
        weight_decay=float(vla["weight_decay"]),
    )
    epochs = args.epochs or (1 if args.smoke else int(vla["epochs"]))
    best_rmse = float("inf")
    best_epoch = 0
    history = []
    started = time.perf_counter()
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for raw_batch in train_loader:
            batch = move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch["images"],
                batch["state_history"],
                batch["instruction_token_ids"],
                batch["instruction_attention_mask"],
            )
            loss = weighted_action_chunk_loss(prediction, batch["target"], batch["weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(vla["gradient_clip_norm"]))
            optimizer.step()
            count = int(prediction.shape[0])
            loss_sum += float(loss.detach()) * count
            sample_count += count
        validation = evaluate(model, validation_loader, device)
        epoch_report = {
            "epoch": epoch,
            "training_weighted_mse": loss_sum / sample_count,
            "validation": validation,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report), flush=True)
        validation_rmse = float(validation["action_chunk_rmse"])
        if validation_rmse < best_rmse:
            best_rmse = validation_rmse
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "vocabulary": vocabulary,
                    "epoch": epoch,
                    "model_type": "hierarchical_vla_preflight",
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    report = {
        "status": "passed",
        "model_type": "hierarchical_vla_preflight",
        "mode": "smoke" if args.smoke else "full",
        "processed_root": str(processed_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "dataset_sizes": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
        "best_epoch": best_epoch,
        "best_validation_action_chunk_rmse": best_rmse,
        "test_metrics": evaluate(model, test_loader, device),
        "checkpoint_sha256": sha256(checkpoint_path),
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
        "claim_boundary": "Preflight imitation model; no closed-loop or open-vocabulary claim.",
        "epochs": history,
    }
    temporary = output_dir / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "report.json")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
