"""Dependency-light sample schema used before introducing CARLA or PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


ROUTE_COMMANDS = {"follow_lane", "left", "right", "straight"}
TRAFFIC_LIGHT_STATES = {"none", "red", "yellow", "green", "unknown"}


class SampleValidationError(ValueError):
    """Raised when a recorded sample violates the data contract."""


@dataclass(frozen=True)
class DrivingSample:
    schema_version: str
    episode_id: str
    route_id: str
    frame_id: int
    timestamp_seconds: float
    rgb_front_path: str
    state: tuple[float, ...]
    traffic_light_state: str
    route_command: str
    expert_action: tuple[float, float]
    collision: bool
    lane_invasion: bool
    red_light_violation: bool
    intervention: bool


def _number_sequence(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SampleValidationError(f"{name} must be a sequence")
    if len(value) != length:
        raise SampleValidationError(f"{name} must contain {length} values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise SampleValidationError(f"{name} must contain only numbers")
    return tuple(float(item) for item in value)


def validate_sample(record: Mapping[str, Any]) -> DrivingSample:
    required = {
        "schema_version",
        "episode_id",
        "route_id",
        "frame_id",
        "timestamp_seconds",
        "observation",
        "expert_action",
        "events",
    }
    missing = required - record.keys()
    if missing:
        raise SampleValidationError(f"missing fields: {sorted(missing)}")
    if record["schema_version"] != "1.0.0":
        raise SampleValidationError("unsupported schema_version")
    if not record["episode_id"] or not record["route_id"]:
        raise SampleValidationError("episode_id and route_id must be non-empty")
    if not isinstance(record["frame_id"], int) or record["frame_id"] < 0:
        raise SampleValidationError("frame_id must be a non-negative integer")
    if not isinstance(record["timestamp_seconds"], (int, float)) or record["timestamp_seconds"] < 0:
        raise SampleValidationError("timestamp_seconds must be non-negative")

    observation = record["observation"]
    if not isinstance(observation, Mapping):
        raise SampleValidationError("observation must be an object")
    state = _number_sequence(observation.get("state"), 8, "observation.state")
    image_path = observation.get("rgb_front_path")
    if not isinstance(image_path, str) or not image_path:
        raise SampleValidationError("rgb_front_path must be non-empty")
    if PurePosixPath(image_path).is_absolute():
        raise SampleValidationError("rgb_front_path must be dataset-relative")
    light = observation.get("traffic_light_state")
    command = observation.get("route_command")
    if light not in TRAFFIC_LIGHT_STATES:
        raise SampleValidationError(f"invalid traffic_light_state: {light}")
    if command not in ROUTE_COMMANDS:
        raise SampleValidationError(f"invalid route_command: {command}")

    action_values = _number_sequence(record["expert_action"], 2, "expert_action")
    if any(value < -1 or value > 1 for value in action_values):
        raise SampleValidationError("expert_action values must be in [-1, 1]")

    events = record["events"]
    event_names = ("collision", "lane_invasion", "red_light_violation", "intervention")
    if not isinstance(events, Mapping) or any(not isinstance(events.get(name), bool) for name in event_names):
        raise SampleValidationError("all event fields must be boolean")

    return DrivingSample(
        schema_version="1.0.0",
        episode_id=str(record["episode_id"]),
        route_id=str(record["route_id"]),
        frame_id=record["frame_id"],
        timestamp_seconds=float(record["timestamp_seconds"]),
        rgb_front_path=image_path,
        state=state,
        traffic_light_state=light,
        route_command=command,
        expert_action=(action_values[0], action_values[1]),
        collision=events["collision"],
        lane_invasion=events["lane_invasion"],
        red_light_violation=events["red_light_violation"],
        intervention=events["intervention"],
    )

