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
