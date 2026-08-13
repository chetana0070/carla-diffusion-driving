"""Dependency-light intervention trigger for corrective demonstration collection."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class InterventionThresholds:
    warmup_ticks: int
    lane_offset_m: float
    heading_error_degrees: float
    unsafe_lead_distance_m: float
    unsafe_lead_min_longitudinal: float
    stall_ticks: int
    stationary_speed_mps: float

    def validate(self) -> None:
        values = (
            self.lane_offset_m,
            self.heading_error_degrees,
            self.unsafe_lead_distance_m,
            self.stationary_speed_mps,
        )
        if self.warmup_ticks < 1 or self.stall_ticks < 1:
            raise ValueError("intervention tick thresholds must be positive")
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("intervention thresholds must be finite and positive")
        if not -1 <= self.unsafe_lead_min_longitudinal <= 1:
            raise ValueError("unsafe lead longitudinal threshold must be in [-1, 1]")


class InterventionMonitor:
    """Detect a pre-failure state without using privileged safety events."""

    def __init__(self, thresholds: InterventionThresholds) -> None:
        thresholds.validate()
        self.thresholds = thresholds
        self.ticks = 0
        self.stationary_ticks = 0

    def update(
        self,
        state: list[float],
        traffic_light_state: str,
        policy_longitudinal: float,
    ) -> tuple[str, ...]:
        if len(state) != 8 or not all(math.isfinite(value) for value in state):
            raise ValueError("intervention state must contain eight finite values")
        if not math.isfinite(policy_longitudinal):
            raise ValueError("policy longitudinal action must be finite")
        self.ticks += 1
        speed = float(state[0])
        lane_offset = float(state[3])
        heading_error_degrees = abs(math.degrees(float(state[4])))
        lead_distance = float(state[5])

        if traffic_light_state == "red" or speed >= self.thresholds.stationary_speed_mps:
            self.stationary_ticks = 0
        else:
            self.stationary_ticks += 1
        if self.ticks < self.thresholds.warmup_ticks:
            return ()

        reasons: list[str] = []
        if abs(lane_offset) >= self.thresholds.lane_offset_m:
            reasons.append("lane_offset")
        if heading_error_degrees >= self.thresholds.heading_error_degrees:
            reasons.append("heading_error")
        if (
            0 <= lead_distance <= self.thresholds.unsafe_lead_distance_m
            and policy_longitudinal >= self.thresholds.unsafe_lead_min_longitudinal
        ):
            reasons.append("unsafe_lead_approach")
        if self.stationary_ticks >= self.thresholds.stall_ticks:
            reasons.append("stalled")
        return tuple(reasons)


@dataclass(frozen=True)
class RecoveryGateThresholds:
    enter_lane_offset_m: float = 0.45
    enter_heading_error_degrees: float = 10.0
    exit_lane_offset_m: float = 0.20
    exit_heading_error_degrees: float = 5.0
    minimum_active_ticks: int = 3
    stable_exit_ticks: int = 3
    maximum_active_ticks: int = 20
    cooldown_ticks: int = 5
    center_crossing_max_abs_offset_m: float = 0.35

    def validate(self) -> None:
        if not 0 < self.exit_lane_offset_m < self.enter_lane_offset_m:
            raise ValueError("lane recovery thresholds must define hysteresis")
        if not 0 < self.exit_heading_error_degrees < self.enter_heading_error_degrees:
            raise ValueError("heading recovery thresholds must define hysteresis")
        if not 0 < self.center_crossing_max_abs_offset_m < self.enter_lane_offset_m:
            raise ValueError("center-crossing threshold must be inside the entry threshold")
        if self.minimum_active_ticks < 1 or self.stable_exit_ticks < 1:
            raise ValueError("recovery timing thresholds must be positive")
        if self.maximum_active_ticks < self.minimum_active_ticks:
            raise ValueError("maximum recovery duration must include minimum duration")
        if self.cooldown_ticks < 0:
            raise ValueError("recovery cooldown cannot be negative")


class RecoveryGate:
    """Activate bounded recovery corrections until the vehicle is stably recentered."""

    def __init__(self, thresholds: RecoveryGateThresholds) -> None:
        thresholds.validate()
        self.thresholds = thresholds
        self.active = False
        self.active_ticks = 0
        self.stable_ticks = 0
        self.cooldown_ticks_remaining = 0
        self.activation_lane_sign = 0
        self.last_transition: str | None = None
        self.last_active_duration_ticks = 0

    def reset(self) -> None:
        self.active = False
        self.active_ticks = 0
        self.stable_ticks = 0
        self.cooldown_ticks_remaining = 0
        self.activation_lane_sign = 0
        self.last_transition = None
        self.last_active_duration_ticks = 0

    def _deactivate(self, reason: str) -> None:
        self.last_active_duration_ticks = self.active_ticks
        self.active = False
        self.active_ticks = 0
        self.stable_ticks = 0
        self.activation_lane_sign = 0
        self.cooldown_ticks_remaining = self.thresholds.cooldown_ticks
        self.last_transition = reason

    def update(self, state: list[float]) -> bool:
        if len(state) != 8 or not all(math.isfinite(value) for value in state):
            raise ValueError("recovery gate state must contain eight finite values")
        signed_lane_offset = float(state[3])
        lane_offset = abs(signed_lane_offset)
        heading_error = abs(math.degrees(float(state[4])))
        self.last_transition = None
        if not self.active:
            if self.cooldown_ticks_remaining > 0:
                self.cooldown_ticks_remaining -= 1
                return False
            if (
                lane_offset >= self.thresholds.enter_lane_offset_m
                or heading_error >= self.thresholds.enter_heading_error_degrees
            ):
                self.active = True
                self.active_ticks = 1
                self.stable_ticks = 0
                self.activation_lane_sign = (
                    1 if signed_lane_offset > 0 else -1 if signed_lane_offset < 0 else 0
                )
                self.last_transition = "activated"
            return self.active

        self.active_ticks += 1
        stable = (
            lane_offset <= self.thresholds.exit_lane_offset_m
            and heading_error <= self.thresholds.exit_heading_error_degrees
        )
        self.stable_ticks = self.stable_ticks + 1 if stable else 0
        crossed_center = (
            self.activation_lane_sign != 0
            and signed_lane_offset * self.activation_lane_sign <= 0
            and lane_offset <= self.thresholds.center_crossing_max_abs_offset_m
        )
        if (
            self.active_ticks >= self.thresholds.minimum_active_ticks
            and crossed_center
        ):
            self._deactivate("center_crossed")
        elif (
            self.active_ticks >= self.thresholds.minimum_active_ticks
            and self.stable_ticks >= self.thresholds.stable_exit_ticks
        ):
            self._deactivate("stably_centered")
        elif self.active_ticks >= self.thresholds.maximum_active_ticks:
            self._deactivate("maximum_duration")
        return self.active


@dataclass(frozen=True)
class RecoverySafetyThresholds:
    preemptive_lane_offset_m: float = 0.15
    preemptive_heading_error_degrees: float = 4.0
    target_speed_mps: float = 4.0
    brake_speed_mps: float = 5.0
    emergency_brake_speed_mps: float = 7.0
    brake_longitudinal: float = -0.35
    emergency_brake_longitudinal: float = -0.70
    direction_guard_lane_offset_m: float = 0.20
    minimum_centering_steering: float = 0.03
    centering_steering_gain: float = 0.25
    maximum_centering_steering: float = 0.15
    maximum_abs_steering: float = 0.45
    maximum_steering_slew_per_tick: float = 0.20

    def validate(self) -> None:
        if not (
            0 < self.target_speed_mps
            < self.brake_speed_mps
            < self.emergency_brake_speed_mps
        ):
            raise ValueError("recovery speed thresholds must be strictly increasing")
        if not -1 <= self.emergency_brake_longitudinal < self.brake_longitudinal < 0:
            raise ValueError("recovery braking actions must be ordered in [-1, 0)")
        if not 0 < self.preemptive_lane_offset_m <= self.direction_guard_lane_offset_m:
            raise ValueError("preemptive lane threshold must precede the direction guard")
        if not 0 < self.preemptive_heading_error_degrees:
            raise ValueError("preemptive heading threshold must be positive")
        if not 0 < self.direction_guard_lane_offset_m:
            raise ValueError("direction guard threshold must be positive")
        if not (
            0
            < self.minimum_centering_steering
            <= self.maximum_centering_steering
            <= self.maximum_abs_steering
        ):
            raise ValueError("proportional centering bounds must fit the steering bound")
        if not math.isfinite(self.centering_steering_gain) or self.centering_steering_gain <= 0:
            raise ValueError("centering steering gain must be finite and positive")
        if not 0 < self.maximum_abs_steering <= 1:
            raise ValueError("maximum recovery steering must be in (0, 1]")
        if not 0 < self.maximum_steering_slew_per_tick <= 2:
            raise ValueError("steering slew limit must be in (0, 2]")


@dataclass(frozen=True)
class RecoverySafetyResult:
    steering: float
    longitudinal: float
    applied_residual_steering: float
    applied_residual_longitudinal: float
    steering_direction_rejected: bool
    final_steering_direction_overridden: bool
    speed_governor_active: bool
    steering_slew_limited: bool


class RecoverySafetyEnvelope:
    """Apply deterministic deployment limits around a learned recovery residual."""

    def __init__(self, thresholds: RecoverySafetyThresholds) -> None:
        thresholds.validate()
        self.thresholds = thresholds
        self.previous_steering: float | None = None

    def reset(self) -> None:
        self.previous_steering = None

    def should_preempt(self, state: list[float]) -> bool:
        """Enter the safety envelope before the learned recovery gate fires."""
        if len(state) != 8 or not all(math.isfinite(value) for value in state):
            raise ValueError("recovery safety state must contain eight finite values")
        return (
            abs(float(state[3])) >= self.thresholds.preemptive_lane_offset_m
            or abs(math.degrees(float(state[4])))
            >= self.thresholds.preemptive_heading_error_degrees
        )

    def _required_centering_steering(self, lane_offset: float) -> float:
        proportional = self.thresholds.centering_steering_gain * abs(lane_offset)
        return min(
            self.thresholds.maximum_centering_steering,
            max(self.thresholds.minimum_centering_steering, proportional),
        )

    def apply(
        self,
        *,
        state: list[float],
        base_steering: float,
        base_longitudinal: float,
        residual_steering: float,
        residual_longitudinal: float,
        safety_active: bool,
    ) -> RecoverySafetyResult:
        values = (
            *state,
            base_steering,
            base_longitudinal,
            residual_steering,
            residual_longitudinal,
        )
        if len(state) != 8 or not all(math.isfinite(value) for value in values):
            raise ValueError("recovery safety inputs must be finite with an eight-value state")
        if not safety_active:
            self.previous_steering = base_steering
            return RecoverySafetyResult(
                steering=base_steering,
                longitudinal=base_longitudinal,
                applied_residual_steering=0.0,
                applied_residual_longitudinal=0.0,
                steering_direction_rejected=False,
                final_steering_direction_overridden=False,
                speed_governor_active=False,
                steering_slew_limited=False,
            )

        speed = float(state[0])
        lane_offset = float(state[3])
        direction_rejected = (
            abs(lane_offset) >= self.thresholds.direction_guard_lane_offset_m
            and lane_offset * residual_steering >= 0
            and residual_steering != 0
        )
        applied_steering = 0.0 if direction_rejected else residual_steering
        # Corrective demonstrations must not add throttle while recovering.
        applied_longitudinal = min(residual_longitudinal, 0.0)

        desired_steering = max(
            -self.thresholds.maximum_abs_steering,
            min(
                self.thresholds.maximum_abs_steering,
                base_steering + applied_steering,
            ),
        )
        center_direction = -1.0 if lane_offset > 0 else 1.0
        required_centering = self._required_centering_steering(lane_offset)
        final_direction_overridden = False
        if (
            abs(lane_offset) >= self.thresholds.direction_guard_lane_offset_m
            and desired_steering * center_direction
            < required_centering
        ):
            desired_steering = center_direction * required_centering
            final_direction_overridden = True
        previous = (
            base_steering if self.previous_steering is None else self.previous_steering
        )
        lower = previous - self.thresholds.maximum_steering_slew_per_tick
        upper = previous + self.thresholds.maximum_steering_slew_per_tick
        steering = max(lower, min(upper, desired_steering))
        slew_limited = not math.isclose(steering, desired_steering, abs_tol=1e-9)
        # A comfort limit must never preserve steering that moves farther from center.
        if (
            abs(lane_offset) >= self.thresholds.direction_guard_lane_offset_m
            and steering * center_direction < required_centering
        ):
            steering = center_direction * required_centering
            final_direction_overridden = True

        longitudinal = max(-1.0, min(1.0, base_longitudinal + applied_longitudinal))
        governed_longitudinal = longitudinal
        if speed >= self.thresholds.emergency_brake_speed_mps:
            governed_longitudinal = min(
                governed_longitudinal,
                self.thresholds.emergency_brake_longitudinal,
            )
        elif speed >= self.thresholds.brake_speed_mps:
            governed_longitudinal = min(
                governed_longitudinal,
                self.thresholds.brake_longitudinal,
            )
        elif speed >= self.thresholds.target_speed_mps:
            governed_longitudinal = min(governed_longitudinal, 0.0)

        self.previous_steering = steering
        return RecoverySafetyResult(
            steering=steering,
            longitudinal=governed_longitudinal,
            applied_residual_steering=applied_steering,
            applied_residual_longitudinal=applied_longitudinal,
            steering_direction_rejected=direction_rejected,
            final_steering_direction_overridden=final_direction_overridden,
            speed_governor_active=not math.isclose(
                longitudinal, governed_longitudinal, abs_tol=1e-9
            ),
            steering_slew_limited=slew_limited,
        )


@dataclass(frozen=True)
class LivenessGuardThresholds:
    stationary_speed_mps: float = 0.10
    release_speed_mps: float = 1.50
    activation_ticks: int = 10
    maximum_active_ticks: int = 30
    cooldown_ticks: int = 20
    minimum_longitudinal: float = 0.30
    minimum_clear_lead_distance_m: float = 8.0
    maximum_abs_lane_offset_m: float = 0.35
    maximum_abs_heading_error_degrees: float = 8.0

    def validate(self) -> None:
        if not 0 < self.stationary_speed_mps < self.release_speed_mps:
            raise ValueError("liveness speed thresholds must be ordered and positive")
        if self.activation_ticks < 1 or self.maximum_active_ticks < 1:
            raise ValueError("liveness timing thresholds must be positive")
        if self.cooldown_ticks < 0:
            raise ValueError("liveness cooldown cannot be negative")
        if not 0 < self.minimum_longitudinal <= 1:
            raise ValueError("minimum liveness longitudinal action must be in (0, 1]")
        geometry = (
            self.minimum_clear_lead_distance_m,
            self.maximum_abs_lane_offset_m,
            self.maximum_abs_heading_error_degrees,
        )
        if not all(math.isfinite(value) and value > 0 for value in geometry):
            raise ValueError("liveness safety geometry must be finite and positive")


@dataclass(frozen=True)
class LivenessGuardResult:
    longitudinal: float
    active: bool
    activated: bool
    exit_reason: str | None
    stationary_ticks: int
    active_ticks: int
    cooldown_ticks_remaining: int


class LivenessGuard:
    """Apply bounded launch authority after a safe, persistent stationary state."""

    def __init__(self, thresholds: LivenessGuardThresholds) -> None:
        thresholds.validate()
        self.thresholds = thresholds
        self.active = False
        self.stationary_ticks = 0
        self.active_ticks = 0
        self.cooldown_ticks_remaining = 0

    def reset(self) -> None:
        self.active = False
        self.stationary_ticks = 0
        self.active_ticks = 0
        self.cooldown_ticks_remaining = 0

    def _lead_is_clear(self, lead_distance_m: float) -> bool:
        return (
            lead_distance_m < 0
            or lead_distance_m >= self.thresholds.minimum_clear_lead_distance_m
        )

    def _geometry_is_safe(self, state: list[float]) -> bool:
        return (
            abs(float(state[3])) <= self.thresholds.maximum_abs_lane_offset_m
            and abs(math.degrees(float(state[4])))
            <= self.thresholds.maximum_abs_heading_error_degrees
        )

    def apply(
        self,
        *,
        state: list[float],
        traffic_light_state: str,
        longitudinal: float,
        safety_active: bool,
    ) -> LivenessGuardResult:
        values = (*state, longitudinal)
        if len(state) != 8 or not all(math.isfinite(value) for value in values):
            raise ValueError("liveness inputs must be finite with an eight-value state")
        speed = float(state[0])
        lead_distance = float(state[5])
        blocked = (
            safety_active
            or traffic_light_state == "red"
            or not self._lead_is_clear(lead_distance)
            or not self._geometry_is_safe(state)
        )
        activated = False
        exit_reason: str | None = None

        if self.active:
            self.active_ticks += 1
            if blocked:
                exit_reason = "safety_blocked"
            elif speed >= self.thresholds.release_speed_mps:
                exit_reason = "moving"
            elif self.active_ticks >= self.thresholds.maximum_active_ticks:
                exit_reason = "maximum_duration"
            if exit_reason is not None:
                self.active = False
                self.active_ticks = 0
                self.stationary_ticks = 0
                self.cooldown_ticks_remaining = self.thresholds.cooldown_ticks
        elif self.cooldown_ticks_remaining > 0:
            self.cooldown_ticks_remaining -= 1
            self.stationary_ticks = 0
        elif blocked or speed > self.thresholds.stationary_speed_mps:
            self.stationary_ticks = 0
        else:
            self.stationary_ticks += 1
            if self.stationary_ticks >= self.thresholds.activation_ticks:
                self.active = True
                self.active_ticks = 1
                activated = True

        governed = (
            max(longitudinal, self.thresholds.minimum_longitudinal)
            if self.active
            else longitudinal
        )
        return LivenessGuardResult(
            longitudinal=governed,
            active=self.active,
            activated=activated,
            exit_reason=exit_reason,
            stationary_ticks=self.stationary_ticks,
            active_ticks=self.active_ticks,
            cooldown_ticks_remaining=self.cooldown_ticks_remaining,
        )


@dataclass(frozen=True)
class DeploymentSpeedGovernorThresholds:
    target_speed_mps: float = 4.0
    brake_speed_mps: float = 4.25
    emergency_brake_speed_mps: float = 4.75
    brake_longitudinal: float = -0.35
    emergency_brake_longitudinal: float = -0.70

    def validate(self) -> None:
        if not (
            0 < self.target_speed_mps
            < self.brake_speed_mps
            < self.emergency_brake_speed_mps
        ):
            raise ValueError("deployment speed thresholds must be strictly increasing")
        if not -1 <= self.emergency_brake_longitudinal < self.brake_longitudinal < 0:
            raise ValueError("deployment braking actions must be ordered in [-1, 0)")


@dataclass(frozen=True)
class DeploymentSpeedGovernorResult:
    longitudinal: float
    active: bool
    mode: str | None


class DeploymentSpeedGovernor:
    """Enforce a global speed envelope independently of lane-recovery state."""

    def __init__(self, thresholds: DeploymentSpeedGovernorThresholds) -> None:
        thresholds.validate()
        self.thresholds = thresholds

    def apply(
        self, *, speed_mps: float, longitudinal: float
    ) -> DeploymentSpeedGovernorResult:
        if not all(math.isfinite(value) for value in (speed_mps, longitudinal)):
            raise ValueError("deployment speed governor inputs must be finite")
        if speed_mps < 0 or not -1 <= longitudinal <= 1:
            raise ValueError("deployment speed and longitudinal action are out of range")

        governed = longitudinal
        mode: str | None = None
        if speed_mps >= self.thresholds.emergency_brake_speed_mps:
            governed = min(governed, self.thresholds.emergency_brake_longitudinal)
            mode = "emergency_brake"
        elif speed_mps >= self.thresholds.brake_speed_mps:
            governed = min(governed, self.thresholds.brake_longitudinal)
            mode = "brake"
        elif speed_mps >= self.thresholds.target_speed_mps:
            governed = min(governed, 0.0)
            mode = "coast"
        return DeploymentSpeedGovernorResult(
            longitudinal=governed,
            active=not math.isclose(governed, longitudinal, abs_tol=1e-9),
            mode=mode,
        )
