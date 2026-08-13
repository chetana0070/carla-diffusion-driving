#!/usr/bin/env python3
"""Evaluate temporal BC with a state-gated bounded recovery residual."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_single_frame_bc_closed_loop as closed_loop_evaluator
from evaluate_temporal_bc_closed_loop import TemporalPolicyRuntime

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
from carla_diffusion.residual_model import ResidualCorrection


class ResidualPolicyRuntime(TemporalPolicyRuntime):
    def __init__(self, checkpoint_path: Path, processed_root: Path) -> None:
        residual_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if residual_checkpoint.get("model_type") != "temporal_bc_gated_residual":
            raise ValueError("checkpoint is not a gated residual policy")
        super().__init__(
            Path(str(residual_checkpoint["base_checkpoint"])), processed_root
        )
        self.residual = ResidualCorrection(
            **residual_checkpoint["model_parameters"]
        ).to(self.device)
        self.residual.load_state_dict(residual_checkpoint["model_state_dict"])
        self.residual.eval()
        self.checkpoint_epoch = int(residual_checkpoint["epoch"])
        self.model_type = "temporal_bc_safety_gated_residual"
        self.gate = RecoveryGate(RecoveryGateThresholds())
        self.safety = RecoverySafetyEnvelope(RecoverySafetyThresholds())
        self.liveness = LivenessGuard(LivenessGuardThresholds())
        self.deployment_speed = DeploymentSpeedGovernor(
            DeploymentSpeedGovernorThresholds()
        )
        self.recovery_active_ticks = 0
        self.safety_active_ticks = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self.gate.reset()
        self.safety.reset()
        self.liveness.reset()
        self.recovery_active_ticks = 0
        self.safety_active_ticks = 0

    @torch.inference_mode()
    def predict(
        self,
        image: Any,
        state: list[float],
        route_command: str,
        traffic_light_state: str,
    ) -> tuple[float, float, float]:
        steering, longitudinal, base_latency_ms = super().predict(
            image, state, route_command, traffic_light_state
        )
        active = self.gate.update(state)
        activated = self.gate.last_transition == "activated"
        preemptive_safety_active = self.safety.should_preempt(state)
        safety_active = (
            active
            or self.gate.cooldown_ticks_remaining > 0
            or preemptive_safety_active
        )
        residual_latency_ms = 0.0
        delta_values = [0.0, 0.0]
        if active:
            self.recovery_active_ticks += 1
            if (
                self.last_state_history is None
                or self.last_condition is None
                or self.last_base_action is None
            ):
                raise RuntimeError("temporal base did not expose residual features")
            torch.cuda.synchronize()
            started = time.perf_counter()
            delta = self.residual(
                self.last_state_history,
                self.last_condition,
                self.last_base_action,
            )
            torch.cuda.synchronize()
            residual_latency_ms = (time.perf_counter() - started) * 1000.0
            delta_values = delta[0].detach().cpu().tolist()
        if safety_active:
            self.safety_active_ticks += 1
        safe_action = self.safety.apply(
            state=state,
            base_steering=steering,
            base_longitudinal=longitudinal,
            residual_steering=float(delta_values[0]),
            residual_longitudinal=float(delta_values[1]),
            safety_active=safety_active,
        )
        # Preemptive steering correction is compatible with a bounded launch.
        # Only active learned recovery or its cooldown owns longitudinal priority.
        liveness_recovery_blocked = (
            active or self.gate.cooldown_ticks_remaining > 0
        )
        liveness = self.liveness.apply(
            state=state,
            traffic_light_state=traffic_light_state,
            longitudinal=safe_action.longitudinal,
            safety_active=liveness_recovery_blocked,
        )
        deployment_speed = self.deployment_speed.apply(
            speed_mps=float(state[0]),
            longitudinal=liveness.longitudinal,
        )
        exit_reasons = {"center_crossed", "stably_centered", "maximum_duration"}
        exit_reason = (
            self.gate.last_transition
            if self.gate.last_transition in exit_reasons
            else None
        )
        self.last_diagnostics = {
            "base_steering": steering,
            "base_longitudinal": longitudinal,
            "residual_steering": float(delta_values[0]),
            "residual_longitudinal": float(delta_values[1]),
            "applied_residual_steering": safe_action.applied_residual_steering,
            "applied_residual_longitudinal": safe_action.applied_residual_longitudinal,
            "recovery_active": active,
            "recovery_activated": activated,
            "recovery_active_ticks": self.recovery_active_ticks,
            "recovery_exit_reason": exit_reason,
            "recovery_gate_transition": self.gate.last_transition,
            "recovery_cooldown_ticks_remaining": self.gate.cooldown_ticks_remaining,
            "safety_active": safety_active,
            "preemptive_safety_active": preemptive_safety_active,
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
            "liveness_cooldown_ticks_remaining": (
                liveness.cooldown_ticks_remaining
            ),
            "liveness_recovery_blocked": liveness_recovery_blocked,
            "deployment_speed_governor_active": deployment_speed.active,
            "deployment_speed_governor_mode": deployment_speed.mode,
        }
        return (
            safe_action.steering,
            deployment_speed.longitudinal,
            base_latency_ms + residual_latency_ms,
        )


def main() -> int:
    closed_loop_evaluator.PolicyRuntime = ResidualPolicyRuntime
    return int(closed_loop_evaluator.main())


if __name__ == "__main__":
    raise SystemExit(main())
