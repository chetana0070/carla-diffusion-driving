"""Configuration loading and cross-field validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the frozen project configuration is internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = json.load(handle)

    required_sections = {
        "project",
        "simulator",
        "camera",
        "policy",
        "dataset",
        "preprocessing",
        "evaluation",
    }
    missing = required_sections - config.keys()
    _require(not missing, f"missing configuration sections: {sorted(missing)}")

    simulator = config["simulator"]
    camera = config["camera"]
    policy = config["policy"]
    dataset = config["dataset"]
    preprocessing = config["preprocessing"]
    evaluation = config["evaluation"]

    _require(simulator["synchronous_mode"] is True, "synchronous_mode must be true")
    _require(simulator["control_hz"] > 0, "control_hz must be positive")
    expected_delta = 1.0 / simulator["control_hz"]
    _require(
        abs(simulator["fixed_delta_seconds"] - expected_delta) < 1e-9,
        "fixed_delta_seconds must equal 1 / control_hz",
    )
    _require(simulator["rpc_port"] != simulator["streaming_port"], "ports must differ")

    _require(camera["history_frames"] >= 2, "temporal policy requires at least two frames")
    _require(camera["capture_width"] >= camera["model_width"], "capture width too small")
    _require(camera["capture_height"] >= camera["model_height"], "capture height too small")
    _require(1 <= camera["jpeg_quality"] <= 100, "jpeg_quality must be in [1, 100]")

    _require(policy["action_dimension"] == 2, "action must be steering + acceleration")
    _require(policy["action_horizon"] >= policy["execute_steps"] > 0, "invalid action horizon")
    _require(dataset["route_level_split"] is True, "frame-level splitting is prohibited")
    _require(
        dataset["stop_collection_free_disk_gib"] < dataset["minimum_free_disk_gib"],
        "collection stop threshold must be below start threshold",
    )
    _require(preprocessing["raw_state_dimension"] == 8, "raw state dimension must be 8")
    _require(
        preprocessing["model_state_dimension"] == 10,
        "model state must add lead/light availability masks",
    )
    _require(
        preprocessing["categorical_condition_dimension"] == 9,
        "condition must encode four route commands and five traffic-light states",
    )
    _require(preprocessing["acceleration_clip_mps2"] > 0, "acceleration clip must be positive")
    _require(
        0 < preprocessing["stationary_hold_weight"] <= 1,
        "stationary hold weight must be in (0, 1]",
    )

    split_sets = [
        set(dataset["train_towns"]),
        set(dataset["validation_towns"]),
        set(dataset["test_towns"]),
    ]
    _require(all(split_sets), "every town split must be non-empty")
    _require(
        not (split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]),
        "town splits must be disjoint",
    )
    _require(len(evaluation["seeds"]) >= 3, "evaluation requires at least three seeds")
    _require(len(set(evaluation["seeds"])) == len(evaluation["seeds"]), "seeds must be unique")
    _require(
        evaluation["max_policy_latency_ms"] <= 1000 / simulator["control_hz"],
        "latency budget exceeds the control period",
    )
    return config
