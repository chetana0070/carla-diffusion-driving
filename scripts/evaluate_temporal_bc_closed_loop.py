#!/usr/bin/env python3
"""Evaluate the four-frame temporal BC checkpoint in the Phase 4 route contract."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_single_frame_bc_closed_loop as closed_loop_evaluator

from carla_diffusion.bc_dataset import build_image_transform
from carla_diffusion.temporal_bc_model import TemporalBC
from carla_diffusion.temporal_runtime import TemporalHistoryBuffer
from carla_diffusion.training_data import categorical_condition, transform_state


class TemporalPolicyRuntime:
    def __init__(self, checkpoint_path: Path, processed_root: Path) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for closed-loop policy evaluation")
        self.device = torch.device("cuda")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.checkpoint_epoch = int(checkpoint["epoch"])
        self.config: dict[str, Any] = checkpoint["config"]
        temporal = self.config["temporal_behavioral_cloning"]
        self.history_frames = int(temporal["history_frames"])
        self.model_type = "temporal_bc"
        self.uses_external_control = False
        self.model = TemporalBC(
            history_frames=self.history_frames,
            state_dimension=int(temporal["state_input_dimension"]),
            condition_dimension=int(temporal["condition_input_dimension"]),
            image_projection_dimension=int(temporal["image_projection_dimension"]),
            state_projection_dimension=int(temporal["state_projection_dimension"]),
            temporal_hidden_dimension=int(temporal["temporal_hidden_dimension"]),
            condition_projection_dimension=int(temporal["condition_projection_dimension"]),
            temporal_layers=int(temporal["temporal_layers"]),
            pretrained=False,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        normalization = json.loads(
            (processed_root / "normalization.json").read_text(encoding="utf-8")
        )
        self.state_mean = torch.tensor(
            normalization["state_mean"], dtype=torch.float32, device=self.device
        )
        self.state_std = torch.tensor(
            normalization["state_std"], dtype=torch.float32, device=self.device
        )
        self.acceleration_clip = float(normalization["acceleration_clip_mps2"])
        self.normalized_state_clip = float(temporal["normalized_state_clip"])
        self.image_transform = build_image_transform(
            int(temporal["image_size"]), augment=False
        )
        self.history: TemporalHistoryBuffer[torch.Tensor, torch.Tensor] = (
            TemporalHistoryBuffer(self.history_frames)
        )
        self.last_state_history: torch.Tensor | None = None
        self.last_condition: torch.Tensor | None = None
        self.last_base_action: torch.Tensor | None = None
        self.warmup_iterations = 20
        image_size = int(temporal["image_size"])
        dummy_images = torch.zeros(
            1,
            self.history_frames,
            3,
            image_size,
            image_size,
            dtype=torch.float32,
            device=self.device,
        )
        dummy_states = torch.zeros(
            1,
            self.history_frames,
            int(temporal["state_input_dimension"]),
            dtype=torch.float32,
            device=self.device,
        )
        dummy_condition = torch.zeros(
            1,
            int(temporal["condition_input_dimension"]),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            for _ in range(self.warmup_iterations):
                self.model(dummy_images, dummy_states, dummy_condition)
        torch.cuda.synchronize()

    def reset_episode(self) -> None:
        self.history.reset()
        self.last_diagnostics: dict[str, Any] = {}

    def diagnostics(self) -> dict[str, Any]:
        return dict(self.last_diagnostics)

    def start_episode(
        self,
        vehicle: Any,
        traffic_manager: Any,
        route: list[Any],
        traffic_manager_port: int,
    ) -> None:
        del vehicle, traffic_manager, route, traffic_manager_port

    @torch.inference_mode()
    def predict(
        self,
        image: Any,
        state: list[float],
        route_command: str,
        traffic_light_state: str,
    ) -> tuple[float, float, float]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        pil_image: Image.Image = closed_loop_evaluator.carla_image_to_pil(image)
        image_tensor = self.image_transform(pil_image).to(self.device)
        transformed_state = torch.tensor(
            transform_state(state, self.acceleration_clip),
            dtype=torch.float32,
            device=self.device,
        )
        normalized_state = torch.clamp(
            (transformed_state - self.state_mean) / self.state_std,
            min=-self.normalized_state_clip,
            max=self.normalized_state_clip,
        )
        self.history.append(image_tensor, normalized_state)
        image_history, state_history = self.history.sequences()
        images = torch.stack(image_history).unsqueeze(0)
        states = torch.stack(state_history).unsqueeze(0)
        condition = torch.tensor(
            categorical_condition(route_command, traffic_light_state),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        prediction = self.model(images, states, condition)
        self.last_state_history = states
        self.last_condition = condition
        self.last_base_action = prediction
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not bool(torch.all(torch.isfinite(prediction))):
            raise FloatingPointError("policy produced a non-finite closed-loop action")
        steering, longitudinal = prediction[0].detach().cpu().tolist()
        self.last_diagnostics = {
            "base_steering": float(steering),
            "base_longitudinal": float(longitudinal),
        }
        return float(steering), float(longitudinal), latency_ms


def main() -> int:
    closed_loop_evaluator.PolicyRuntime = TemporalPolicyRuntime
    return int(closed_loop_evaluator.main())


if __name__ == "__main__":
    raise SystemExit(main())
