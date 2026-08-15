#!/usr/bin/env python3
"""Evaluate the promoted factorized policy with frozen Phase 6 arbitration."""

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
from train_factorized_policy import build_model

from carla_diffusion.bc_dataset import build_image_transform
from carla_diffusion.diffusion_policy import DiffusionSchedule
from carla_diffusion.factorized_runtime import (
    ActionChunkExecutor,
    deployment_noise_seed,
)
from carla_diffusion.intervention import (
    DeploymentSpeedGovernor,
    DeploymentSpeedGovernorThresholds,
    LivenessGuard,
    LivenessGuardThresholds,
    RecoveryGate,
    RecoveryGateThresholds,
    RecoverySafetyEnvelope,
    RecoverySafetyThresholds,
)
from carla_diffusion.temporal_runtime import TemporalHistoryBuffer
from carla_diffusion.training_data import categorical_condition, transform_state


class FactorizedPolicyRuntime:
    """Replan bounded action chunks and arbitrate every executed control tick."""

    def __init__(self, checkpoint_path: Path, processed_root: Path) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for factorized closed-loop evaluation")
        self.device = torch.device("cuda")
        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        if checkpoint.get("model_type") != "factorized_temporal_policy":
            raise ValueError("checkpoint is not a factorized temporal policy")
        self.checkpoint_epoch = int(checkpoint["epoch"])
        self.config: dict[str, Any] = checkpoint["config"]
        factorized = self.config["factorized_policy"]
        self.history_frames = int(factorized["history_frames"])
        self.model_type = "factorized_temporal_policy_phase6_safety_arbitration"
        self.uses_external_control = False
        self.model = build_model(factorized, pretrained=False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.schedule = DiffusionSchedule(
            int(factorized["diffusion_steps"]), float(factorized["cosine_s"])
        ).to(self.device)
        self.inference_steps = int(factorized["inference_steps"])
        self.ddim_eta = float(factorized["ddim_eta"])
        self.action_horizon = int(factorized["action_horizon"])
        self.action_dimension = int(factorized["action_dimension"])
        self.execute_steps = int(factorized["execute_steps"])
        self.noise_seed = deployment_noise_seed(self.config)
        self.generator = torch.Generator(device=self.device)
        self.chunk = ActionChunkExecutor(self.execute_steps)
        self.history: TemporalHistoryBuffer[torch.Tensor, torch.Tensor] = (
            TemporalHistoryBuffer(self.history_frames)
        )
        self.last_context_key: tuple[str, str] | None = None
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
        self.normalized_state_clip = float(factorized["normalized_state_clip"])
        self.image_transform = build_image_transform(
            int(factorized["image_size"]), augment=False
        )
        self.gate = RecoveryGate(RecoveryGateThresholds())
        self.safety = RecoverySafetyEnvelope(RecoverySafetyThresholds())
        self.liveness = LivenessGuard(LivenessGuardThresholds())
        self.deployment_speed = DeploymentSpeedGovernor(
            DeploymentSpeedGovernorThresholds()
        )
        self.last_diagnostics: dict[str, Any] = {}
        self.replan_count = 0
        self.recovery_active_ticks = 0
        self.safety_active_ticks = 0
        self.warmup_iterations = 10
        image_size = int(factorized["image_size"])
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
            int(factorized["state_input_dimension"]),
            dtype=torch.float32,
            device=self.device,
        )
        dummy_condition = torch.zeros(
            1,
            int(factorized["condition_input_dimension"]),
            dtype=torch.float32,
            device=self.device,
        )
        dummy_noise = torch.zeros(
            1,
            self.action_horizon,
            self.action_dimension,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            for _ in range(self.warmup_iterations):
                self.model.sample(
                    self.schedule,
                    dummy_images,
                    dummy_states,
                    dummy_condition,
                    inference_steps=self.inference_steps,
                    eta=self.ddim_eta,
                    initial_noise=dummy_noise,
                )
        torch.cuda.synchronize()
        self.reset_episode()

    def reset_episode(self) -> None:
        self.history.reset()
        self.chunk.reset()
        self.gate.reset()
        self.safety.reset()
        self.liveness.reset()
        self.last_context_key = None
        self.generator.manual_seed(self.noise_seed)
        self.last_diagnostics = {}
        self.replan_count = 0
        self.recovery_active_ticks = 0
        self.safety_active_ticks = 0

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
        context_key = (route_command, traffic_light_state)
        context_changed = (
            self.last_context_key is not None and context_key != self.last_context_key
        )
        if context_changed:
            self.chunk.reset()
        self.last_context_key = context_key
        replanned = self.chunk.needs_replan
        plan_step = self.chunk.step_index
        if replanned:
            image_history, state_history = self.history.sequences()
            images = torch.stack(image_history).unsqueeze(0)
            states = torch.stack(state_history).unsqueeze(0)
            condition = torch.tensor(
                categorical_condition(route_command, traffic_light_state),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            initial_noise = torch.randn(
                1,
                self.action_horizon,
                self.action_dimension,
                generator=self.generator,
                device=self.device,
            )
            prediction = self.model.sample(
                self.schedule,
                images,
                states,
                condition,
                inference_steps=self.inference_steps,
                eta=self.ddim_eta,
                initial_noise=initial_noise,
            )
            if not bool(torch.all(torch.isfinite(prediction))):
                raise FloatingPointError("factorized policy produced a non-finite chunk")
            self.chunk.set_plan(prediction[0].detach().cpu().tolist())
            self.replan_count += 1
            plan_step = 0
        base_steering, base_longitudinal = self.chunk.pop()

        active = self.gate.update(state)
        activated = self.gate.last_transition == "activated"
        preemptive = self.safety.should_preempt(state)
        safety_active = active or self.gate.cooldown_ticks_remaining > 0 or preemptive
        if active:
            self.recovery_active_ticks += 1
        if safety_active:
            self.safety_active_ticks += 1
        safe_action = self.safety.apply(
            state=state,
            base_steering=base_steering,
            base_longitudinal=base_longitudinal,
            residual_steering=0.0,
            residual_longitudinal=0.0,
            safety_active=safety_active,
        )
        recovery_blocked = active or self.gate.cooldown_ticks_remaining > 0
        liveness = self.liveness.apply(
            state=state,
            traffic_light_state=traffic_light_state,
            longitudinal=safe_action.longitudinal,
            safety_active=recovery_blocked,
        )
        deployment_speed = self.deployment_speed.apply(
            speed_mps=float(state[0]), longitudinal=liveness.longitudinal
        )
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        exit_reasons = {"center_crossed", "stably_centered", "maximum_duration"}
        recovery_exit = (
            self.gate.last_transition
            if self.gate.last_transition in exit_reasons
            else None
        )
        self.last_diagnostics = {
            "base_steering": base_steering,
            "base_longitudinal": base_longitudinal,
            "factorized_policy_replanned": replanned,
            "factorized_plan_step": plan_step,
            "factorized_replan_count": self.replan_count,
            "factorized_context_change_replan": context_changed,
            "factorized_noise_seed": self.noise_seed,
            "factorized_remaining_plan_steps": self.chunk.remaining_steps,
            "recovery_active": active,
            "recovery_activated": activated,
            "recovery_active_ticks": self.recovery_active_ticks,
            "recovery_exit_reason": recovery_exit,
            "recovery_gate_transition": self.gate.last_transition,
            "recovery_cooldown_ticks_remaining": self.gate.cooldown_ticks_remaining,
            "safety_active": safety_active,
            "preemptive_safety_active": preemptive,
            "safety_active_ticks": self.safety_active_ticks,
            "steering_direction_rejected": safe_action.steering_direction_rejected,
            "final_steering_direction_overridden": (
                safe_action.final_steering_direction_overridden
            ),
            "speed_governor_active": safe_action.speed_governor_active,
            "steering_slew_limited": safe_action.steering_slew_limited,
            "liveness_active": liveness.active,
            "liveness_activated": liveness.activated,
            "liveness_exit_reason": liveness.exit_reason,
            "liveness_stationary_ticks": liveness.stationary_ticks,
            "liveness_active_ticks": liveness.active_ticks,
            "liveness_cooldown_ticks_remaining": liveness.cooldown_ticks_remaining,
            "liveness_recovery_blocked": recovery_blocked,
            "deployment_speed_governor_active": deployment_speed.active,
            "deployment_speed_governor_mode": deployment_speed.mode,
        }
        return safe_action.steering, deployment_speed.longitudinal, latency_ms


def main() -> int:
    closed_loop_evaluator.PolicyRuntime = FactorizedPolicyRuntime
    return int(closed_loop_evaluator.main())


if __name__ == "__main__":
    raise SystemExit(main())
