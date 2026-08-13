import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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

try:
    import torch

    from carla_diffusion.residual_model import ResidualCorrection, corrected_action
except ImportError:
    torch = None


def state(lane_offset: float, heading_degrees: float) -> list[float]:
    return [2.0, 0.0, 8.0, lane_offset, math.radians(heading_degrees), -1.0, 0.0, -1.0]


class RecoveryGateTests(unittest.TestCase):
    def test_hysteresis_requires_stable_recovery(self) -> None:
        gate = RecoveryGate(
            RecoveryGateThresholds(minimum_active_ticks=2, stable_exit_ticks=2)
        )
        self.assertFalse(gate.update(state(0.1, 1.0)))
        self.assertTrue(gate.update(state(0.5, 1.0)))
        self.assertTrue(gate.update(state(0.1, 1.0)))
        self.assertFalse(gate.update(state(0.1, 1.0)))
        self.assertFalse(gate.update(state(0.1, 1.0)))

    def test_center_crossing_exits_and_starts_cooldown(self) -> None:
        gate = RecoveryGate(
            RecoveryGateThresholds(
                minimum_active_ticks=3,
                stable_exit_ticks=3,
                maximum_active_ticks=12,
                cooldown_ticks=2,
            )
        )
        self.assertTrue(gate.update(state(0.5, 8.0)))
        self.assertTrue(gate.update(state(0.2, 8.0)))
        self.assertFalse(gate.update(state(-0.1, 8.0)))
        self.assertEqual(gate.last_transition, "center_crossed")
        self.assertEqual(gate.cooldown_ticks_remaining, 2)
        self.assertFalse(gate.update(state(0.6, 12.0)))
        self.assertFalse(gate.update(state(0.6, 12.0)))
        self.assertTrue(gate.update(state(0.6, 12.0)))

    def test_maximum_duration_stops_recovery(self) -> None:
        gate = RecoveryGate(
            RecoveryGateThresholds(
                minimum_active_ticks=2,
                stable_exit_ticks=2,
                maximum_active_ticks=4,
            )
        )
        self.assertTrue(gate.update(state(0.6, 12.0)))
        self.assertTrue(gate.update(state(0.6, 12.0)))
        self.assertTrue(gate.update(state(0.6, 12.0)))
        self.assertFalse(gate.update(state(0.6, 12.0)))
        self.assertEqual(gate.last_transition, "maximum_duration")
        self.assertEqual(gate.last_active_duration_ticks, 4)


class RecoverySafetyEnvelopeTests(unittest.TestCase):
    def test_preempts_before_recovery_gate_threshold(self) -> None:
        envelope = RecoverySafetyEnvelope(RecoverySafetyThresholds())
        self.assertFalse(envelope.should_preempt(state(0.14, 3.9)))
        self.assertTrue(envelope.should_preempt(state(0.15, 3.9)))
        self.assertTrue(envelope.should_preempt(state(0.0, 4.0)))

    def test_preemptive_state_validation(self) -> None:
        envelope = RecoverySafetyEnvelope(RecoverySafetyThresholds())
        with self.assertRaises(ValueError):
            envelope.should_preempt([0.0] * 7)

    def test_rejects_steering_that_increases_signed_lane_error(self) -> None:
        envelope = RecoverySafetyEnvelope(RecoverySafetyThresholds())
        result = envelope.apply(
            state=state(0.6, 3.0),
            base_steering=0.1,
            base_longitudinal=0.2,
            residual_steering=0.4,
            residual_longitudinal=-0.1,
            safety_active=True,
        )
        self.assertTrue(result.steering_direction_rejected)
        self.assertEqual(result.applied_residual_steering, 0.0)

    def test_suppresses_positive_residual_throttle_and_governs_speed(self) -> None:
        envelope = RecoverySafetyEnvelope(RecoverySafetyThresholds())
        result = envelope.apply(
            state=[5.5, *state(0.6, 3.0)[1:]],
            base_steering=0.0,
            base_longitudinal=0.7,
            residual_steering=-0.2,
            residual_longitudinal=0.4,
            safety_active=True,
        )
        self.assertEqual(result.applied_residual_longitudinal, 0.0)
        self.assertEqual(result.longitudinal, -0.35)
        self.assertTrue(result.speed_governor_active)

    def test_emergency_speed_uses_stronger_braking(self) -> None:
        envelope = RecoverySafetyEnvelope(RecoverySafetyThresholds())
        result = envelope.apply(
            state=[7.5, *state(0.6, 3.0)[1:]],
            base_steering=0.0,
            base_longitudinal=0.4,
            residual_steering=-0.2,
            residual_longitudinal=-0.1,
            safety_active=True,
        )
        self.assertEqual(result.longitudinal, -0.70)

    def test_limits_steering_slew(self) -> None:
        envelope = RecoverySafetyEnvelope(
            RecoverySafetyThresholds(maximum_steering_slew_per_tick=0.2)
        )
        envelope.apply(
            state=state(0.1, 1.0),
            base_steering=0.2,
            base_longitudinal=0.0,
            residual_steering=0.0,
            residual_longitudinal=0.0,
            safety_active=False,
        )
        result = envelope.apply(
            state=state(0.6, 3.0),
            base_steering=0.2,
            base_longitudinal=0.0,
            residual_steering=-0.5,
            residual_longitudinal=0.0,
            safety_active=True,
        )
        self.assertAlmostEqual(result.steering, -0.15)
        self.assertTrue(result.steering_slew_limited)
        self.assertTrue(result.final_steering_direction_overridden)

    def test_final_action_overrides_base_that_cancels_residual(self) -> None:
        envelope = RecoverySafetyEnvelope(RecoverySafetyThresholds())
        result = envelope.apply(
            state=state(0.6, 3.0),
            base_steering=0.46,
            base_longitudinal=0.2,
            residual_steering=-0.49,
            residual_longitudinal=0.0,
            safety_active=True,
        )
        self.assertAlmostEqual(result.steering, -0.15)
        self.assertTrue(result.final_steering_direction_overridden)

    def test_final_action_centers_negative_lane_offset(self) -> None:
        envelope = RecoverySafetyEnvelope(RecoverySafetyThresholds())
        result = envelope.apply(
            state=state(-0.6, 3.0),
            base_steering=-0.4,
            base_longitudinal=0.2,
            residual_steering=0.45,
            residual_longitudinal=0.0,
            safety_active=True,
        )
        self.assertAlmostEqual(result.steering, 0.15)

    def test_centering_authority_scales_with_lane_error(self) -> None:
        envelope = RecoverySafetyEnvelope(RecoverySafetyThresholds())
        result = envelope.apply(
            state=state(0.24, 3.0),
            base_steering=0.2,
            base_longitudinal=0.2,
            residual_steering=0.0,
            residual_longitudinal=0.0,
            safety_active=True,
        )
        self.assertAlmostEqual(result.steering, -0.06)


class LivenessGuardTests(unittest.TestCase):
    def test_applies_launch_floor_after_persistent_safe_stall(self) -> None:
        guard = LivenessGuard(
            LivenessGuardThresholds(activation_ticks=3, minimum_longitudinal=0.3)
        )
        current_state = state(0.1, 1.0)
        current_state[0] = 0.01
        first = guard.apply(
            state=current_state,
            traffic_light_state="none",
            longitudinal=0.02,
            safety_active=False,
        )
        second = guard.apply(
            state=current_state,
            traffic_light_state="none",
            longitudinal=0.02,
            safety_active=False,
        )
        third = guard.apply(
            state=current_state,
            traffic_light_state="none",
            longitudinal=0.02,
            safety_active=False,
        )
        self.assertFalse(first.active)
        self.assertFalse(second.active)
        self.assertTrue(third.active)
        self.assertTrue(third.activated)
        self.assertEqual(third.longitudinal, 0.3)

    def test_red_light_and_close_lead_block_activation(self) -> None:
        guard = LivenessGuard(LivenessGuardThresholds(activation_ticks=1))
        current_state = state(0.1, 1.0)
        current_state[0] = 0.0
        red = guard.apply(
            state=current_state,
            traffic_light_state="red",
            longitudinal=0.0,
            safety_active=False,
        )
        self.assertFalse(red.active)
        current_state[5] = 3.0
        lead = guard.apply(
            state=current_state,
            traffic_light_state="none",
            longitudinal=0.0,
            safety_active=False,
        )
        self.assertFalse(lead.active)

    def test_active_guard_exits_when_vehicle_moves(self) -> None:
        guard = LivenessGuard(
            LivenessGuardThresholds(activation_ticks=1, release_speed_mps=1.0)
        )
        current_state = state(0.1, 1.0)
        current_state[0] = 0.0
        started = guard.apply(
            state=current_state,
            traffic_light_state="none",
            longitudinal=0.0,
            safety_active=False,
        )
        self.assertTrue(started.active)
        current_state[0] = 1.1
        released = guard.apply(
            state=current_state,
            traffic_light_state="none",
            longitudinal=0.0,
            safety_active=False,
        )
        self.assertFalse(released.active)
        self.assertEqual(released.exit_reason, "moving")
        self.assertEqual(released.longitudinal, 0.0)

    def test_safety_envelope_preempts_active_liveness_guard(self) -> None:
        guard = LivenessGuard(LivenessGuardThresholds(activation_ticks=1))
        current_state = state(0.1, 1.0)
        current_state[0] = 0.0
        guard.apply(
            state=current_state,
            traffic_light_state="none",
            longitudinal=0.0,
            safety_active=False,
        )
        blocked = guard.apply(
            state=current_state,
            traffic_light_state="none",
            longitudinal=-0.35,
            safety_active=True,
        )
        self.assertFalse(blocked.active)
        self.assertEqual(blocked.exit_reason, "safety_blocked")
        self.assertEqual(blocked.longitudinal, -0.35)

    def test_safe_preemptive_lane_geometry_allows_launch(self) -> None:
        guard = LivenessGuard(LivenessGuardThresholds(activation_ticks=1))
        current_state = state(0.24, 1.0)
        current_state[0] = 0.0
        result = guard.apply(
            state=current_state,
            traffic_light_state="none",
            longitudinal=-0.1,
            safety_active=False,
        )
        self.assertTrue(result.active)
        self.assertEqual(result.longitudinal, 0.3)


class DeploymentSpeedGovernorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.governor = DeploymentSpeedGovernor(
            DeploymentSpeedGovernorThresholds()
        )

    def test_below_target_preserves_action(self) -> None:
        result = self.governor.apply(speed_mps=3.9, longitudinal=0.8)
        self.assertFalse(result.active)
        self.assertIsNone(result.mode)
        self.assertEqual(result.longitudinal, 0.8)

    def test_coasts_at_target_speed(self) -> None:
        result = self.governor.apply(speed_mps=4.0, longitudinal=0.8)
        self.assertTrue(result.active)
        self.assertEqual(result.mode, "coast")
        self.assertEqual(result.longitudinal, 0.0)

    def test_brakes_before_the_acceptance_ceiling(self) -> None:
        result = self.governor.apply(speed_mps=4.3, longitudinal=0.8)
        self.assertEqual(result.mode, "brake")
        self.assertEqual(result.longitudinal, -0.35)

    def test_emergency_brakes_with_step_margin(self) -> None:
        result = self.governor.apply(speed_mps=4.8, longitudinal=0.8)
        self.assertEqual(result.mode, "emergency_brake")
        self.assertEqual(result.longitudinal, -0.70)

    def test_stationary_liveness_launch_is_unchanged(self) -> None:
        result = self.governor.apply(speed_mps=0.0, longitudinal=0.3)
        self.assertEqual(result.longitudinal, 0.3)


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class ResidualModelTests(unittest.TestCase):
    def test_residual_is_bounded_and_action_is_clipped(self) -> None:
        assert torch is not None
        model = ResidualCorrection()
        states = torch.zeros(3, 4, 10)
        condition = torch.zeros(3, 9)
        base = torch.tensor([[0.9, -0.9]] * 3)
        delta = model(states, condition, base)
        self.assertEqual(tuple(delta.shape), (3, 2))
        self.assertTrue(bool(torch.all(torch.abs(delta) <= 0.5)))
        action = corrected_action(base, delta)
        self.assertTrue(bool(torch.all(torch.abs(action) <= 1.0)))


if __name__ == "__main__":
    unittest.main()
