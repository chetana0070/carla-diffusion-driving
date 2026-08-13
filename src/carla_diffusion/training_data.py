"""Leakage-resistant temporal-window preparation for policy training."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

from .schema import DrivingSample

ROUTE_COMMAND_ORDER = ("follow_lane", "left", "right", "straight")
TRAFFIC_LIGHT_ORDER = ("none", "red", "yellow", "green", "unknown")


@dataclass(frozen=True)
class WeightConfig:
    stationary_hold: float = 0.25
    turning: float = 2.0
    active_braking: float = 2.0
    lead_context: float = 1.5


@dataclass(frozen=True)
class PreparedWindow:
    split: str
    episode_id: str
    route_id: str
    anchor_frame_id: int
    image_paths: tuple[str, ...]
    state_history: tuple[tuple[float, ...], ...]
    condition: tuple[float, ...]
    action_target: tuple[tuple[float, float], ...]
    sample_weight: float
    categories: tuple[str, ...]
    source_dataset: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.source_dataset is None:
            payload.pop("source_dataset")
        return payload


def transform_state(state: Sequence[float], acceleration_clip: float) -> tuple[float, ...]:
    """Clip acceleration and replace missing distances with explicit availability masks."""
    if len(state) != 8:
        raise ValueError("raw state must have eight elements")
    if acceleration_clip <= 0:
        raise ValueError("acceleration_clip must be positive")
    values = tuple(float(value) for value in state)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("state values must be finite")
    lead_available = values[5] >= 0.0
    light_available = values[7] >= 0.0
    return (
        values[0],
        max(-acceleration_clip, min(acceleration_clip, values[1])),
        values[2],
        values[3],
        values[4],
        values[5] if lead_available else 0.0,
        values[6] if lead_available else 0.0,
        values[7] if light_available else 0.0,
        float(lead_available),
        float(light_available),
    )


def categorical_condition(route_command: str, traffic_light_state: str) -> tuple[float, ...]:
    if route_command not in ROUTE_COMMAND_ORDER:
        raise ValueError(f"unknown route command: {route_command}")
    if traffic_light_state not in TRAFFIC_LIGHT_ORDER:
        raise ValueError(f"unknown traffic-light state: {traffic_light_state}")
    return tuple(float(route_command == name) for name in ROUTE_COMMAND_ORDER) + tuple(
        float(traffic_light_state == name) for name in TRAFFIC_LIGHT_ORDER
    )


def window_weight(sample: DrivingSample, config: WeightConfig) -> tuple[float, tuple[str, ...]]:
    speed = sample.state[0]
    steering, longitudinal = sample.expert_action
    stationary_hold = speed < 0.5 and longitudinal < -0.05
    categories: list[str] = []
    if stationary_hold:
        return config.stationary_hold, ("stationary_brake_hold",)

    weight = 1.0
    if abs(steering) >= 0.05:
        categories.append("turning")
        weight = max(weight, config.turning)
    if speed >= 0.5 and longitudinal < -0.05:
        categories.append("active_braking")
        weight = max(weight, config.active_braking)
    if sample.state[5] >= 0.0:
        categories.append("lead_context")
        weight = max(weight, config.lead_context)
    if not categories:
        categories.append("routine")
    return weight, tuple(categories)


def build_temporal_windows(
    samples: Sequence[DrivingSample],
    *,
    split: str,
    history: int,
    horizon: int,
    acceleration_clip: float,
    weight_config: WeightConfig,
    source_dataset: str | None = None,
) -> list[PreparedWindow]:
    if history < 1 or horizon < 1:
        raise ValueError("history and horizon must be positive")
    if not samples:
        return []
    episode_ids = {sample.episode_id for sample in samples}
    route_ids = {sample.route_id for sample in samples}
    if len(episode_ids) != 1 or len(route_ids) != 1:
        raise ValueError("one episode and one route are required per window sequence")
    ordered = sorted(samples, key=lambda sample: sample.frame_id)
    if any(current.frame_id <= previous.frame_id for previous, current in pairwise(ordered)):
        raise ValueError("frame IDs must be strictly increasing")

    windows: list[PreparedWindow] = []
    for anchor_index in range(history - 1, len(ordered) - horizon + 1):
        anchor = ordered[anchor_index]
        observation_slice = ordered[anchor_index - history + 1 : anchor_index + 1]
        action_slice = ordered[anchor_index : anchor_index + horizon]
        weight, categories = window_weight(anchor, weight_config)
        windows.append(
            PreparedWindow(
                split=split,
                episode_id=anchor.episode_id,
                route_id=anchor.route_id,
                anchor_frame_id=anchor.frame_id,
                image_paths=tuple(sample.rgb_front_path for sample in observation_slice),
                state_history=tuple(
                    transform_state(sample.state, acceleration_clip)
                    for sample in observation_slice
                ),
                condition=categorical_condition(
                    anchor.route_command,
                    anchor.traffic_light_state,
                ),
                action_target=tuple(sample.expert_action for sample in action_slice),
                sample_weight=weight,
                categories=categories,
                source_dataset=source_dataset,
            )
        )
    return windows


def normalization_statistics(
    samples: Iterable[DrivingSample], acceleration_clip: float
) -> dict[str, Any]:
    transformed = [transform_state(sample.state, acceleration_clip) for sample in samples]
    if not transformed:
        raise ValueError("normalization requires at least one training sample")
    feature_count = len(transformed[0])
    means = [sum(row[index] for row in transformed) / len(transformed) for index in range(feature_count)]
    standard_deviations = []
    for index, mean in enumerate(means):
        variance = sum((row[index] - mean) ** 2 for row in transformed) / len(transformed)
        standard_deviations.append(max(math.sqrt(variance), 1e-6))
    # Availability masks stay binary and are never standardized.
    for index in (8, 9):
        means[index] = 0.0
        standard_deviations[index] = 1.0
    return {
        "state_mean": means,
        "state_std": standard_deviations,
        "normalize_indices": list(range(8)),
        "mask_indices": [8, 9],
        "acceleration_clip_mps2": acceleration_clip,
        "training_samples": len(transformed),
    }
