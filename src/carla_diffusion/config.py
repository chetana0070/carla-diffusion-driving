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
        "behavioral_cloning",
        "closed_loop_evaluation",
        "corrective_collection",
        "temporal_behavioral_cloning",
        "diffusion_policy",
        "diffusion_offline_evaluation",
        "factorized_policy",
        "factorized_offline_evaluation",
        "factorized_longitudinal_finetuning",
        "phase7_closed_loop_evaluation",
        "vision_language_action",
        "vla_offline_evaluation",
        "evaluation",
    }
    missing = required_sections - config.keys()
    _require(not missing, f"missing configuration sections: {sorted(missing)}")

    simulator = config["simulator"]
    camera = config["camera"]
    policy = config["policy"]
    dataset = config["dataset"]
    preprocessing = config["preprocessing"]
    behavioral_cloning = config["behavioral_cloning"]
    closed_loop = config["closed_loop_evaluation"]
    corrective = config["corrective_collection"]
    temporal_bc = config["temporal_behavioral_cloning"]
    diffusion = config["diffusion_policy"]
    diffusion_evaluation = config["diffusion_offline_evaluation"]
    factorized = config["factorized_policy"]
    factorized_evaluation = config["factorized_offline_evaluation"]
    factorized_finetuning = config["factorized_longitudinal_finetuning"]
    phase7_closed_loop = config["phase7_closed_loop_evaluation"]
    vla = config["vision_language_action"]
    vla_evaluation = config["vla_offline_evaluation"]
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
    _require(
        behavioral_cloning["image_encoder"] == policy["visual_encoder"],
        "BC and policy encoders must match",
    )
    expected_scalar_dimension = (
        preprocessing["model_state_dimension"]
        + preprocessing["categorical_condition_dimension"]
    )
    _require(
        behavioral_cloning["scalar_input_dimension"] == expected_scalar_dimension,
        "BC scalar input must equal model state plus categorical condition",
    )
    _require(
        behavioral_cloning["output_dimension"] == policy["action_dimension"],
        "BC output must match the action dimension",
    )
    _require(
        behavioral_cloning["image_size"] == camera["model_width"]
        == camera["model_height"],
        "BC image size must match the square model input",
    )
    _require(behavioral_cloning["batch_size"] > 0, "BC batch size must be positive")
    _require(behavioral_cloning["epochs"] > 0, "BC epochs must be positive")
    _require(
        behavioral_cloning["freeze_encoder_epochs"] < behavioral_cloning["epochs"],
        "encoder freeze period must be shorter than training",
    )
    _require(
        0 < behavioral_cloning["learning_rate"] < 1,
        "BC learning rate must be in (0, 1)",
    )
    _require(
        behavioral_cloning["normalized_state_clip"] > 0,
        "normalized-state clip must be positive",
    )
    _require(closed_loop["episodes"] > 0, "closed-loop episodes must be positive")
    _require(
        closed_loop["ticks_per_episode"] > 0,
        "closed-loop ticks per episode must be positive",
    )
    _require(closed_loop["route_points"] >= 20, "closed-loop route is too short")
    _require(closed_loop["route_spacing_m"] > 0, "route spacing must be positive")
    _require(
        closed_loop["command_lookahead_points"] > 0,
        "route-command lookahead must be positive",
    )
    _require(
        closed_loop["no_progress_window_ticks"] >= simulator["control_hz"],
        "no-progress window must span at least one second",
    )
    _require(
        closed_loop["no_progress_min_distance_m"] > 0,
        "no-progress distance must be positive",
    )
    _require(
        0 < closed_loop["expert_oracle_min_route_progress_fraction"] <= 1,
        "expert-oracle route progress must be in (0, 1]",
    )
    _require(
        closed_loop["expert_oracle_protocol_version"] == "2.0.0",
        "unsupported expert-oracle protocol version",
    )
    _require(
        closed_loop["expert_oracle_min_mean_distance_m"] > 0,
        "expert-oracle mean distance must be positive",
    )
    _require(
        closed_loop["expert_oracle_max_collisions"] >= 0,
        "expert-oracle collision allowance must be non-negative",
    )
    _require(
        closed_loop["expert_oracle_max_lane_invasions"] >= 0,
        "expert-oracle lane-invasion allowance must be non-negative",
    )
    _require(
        closed_loop["expert_oracle_max_red_light_violations"] >= 0,
        "expert-oracle red-light allowance must be non-negative",
    )
    _require(
        closed_loop["expert_oracle_max_no_progress_terminations"] >= 0,
        "expert-oracle no-progress allowance must be non-negative",
    )
    _require(corrective["episodes"] > 0, "corrective episodes must be positive")
    _require(
        corrective["minimum_free_disk_gib"]
        > corrective["stop_collection_free_disk_gib"]
        > 0,
        "invalid corrective storage thresholds",
    )
    _require(
        corrective["ticks_per_episode"] >= corrective["expert_recovery_ticks"],
        "corrective rollout must fit the recovery segment",
    )
    _require(
        corrective["minimum_published_samples"]
        >= camera["history_frames"] + policy["action_horizon"],
        "corrective episodes must support at least one temporal window",
    )
    _require(
        corrective["expert_recovery_ticks"] >= corrective["minimum_published_samples"],
        "expert recovery must satisfy the publication minimum",
    )
    _require(
        0 < corrective["minimum_intervention_episodes"] <= corrective["episodes"],
        "invalid corrective intervention episode gate",
    )
    _require(
        corrective["lane_offset_trigger_m"] > 0
        and corrective["heading_error_trigger_degrees"] > 0,
        "corrective geometry thresholds must be positive",
    )
    _require(
        temporal_bc["history_frames"] == camera["history_frames"],
        "temporal BC history must match the observation contract",
    )
    _require(
        temporal_bc["image_encoder"] == policy["visual_encoder"],
        "temporal BC and policy encoders must match",
    )
    _require(
        temporal_bc["state_input_dimension"] == preprocessing["model_state_dimension"],
        "temporal BC state dimension must match preprocessing",
    )
    _require(
        temporal_bc["condition_input_dimension"]
        == preprocessing["categorical_condition_dimension"],
        "temporal BC condition dimension must match preprocessing",
    )
    _require(
        temporal_bc["output_dimension"] == policy["action_dimension"],
        "temporal BC output must match the action dimension",
    )
    _require(
        temporal_bc["image_size"] == camera["model_width"] == camera["model_height"],
        "temporal BC image size must match the square model input",
    )
    _require(temporal_bc["batch_size"] > 0, "temporal BC batch size must be positive")
    _require(temporal_bc["epochs"] > 0, "temporal BC epochs must be positive")
    _require(
        temporal_bc["freeze_encoder_epochs"] < temporal_bc["epochs"],
        "temporal encoder freeze period must be shorter than training",
    )
    _require(
        temporal_bc["normalized_state_clip"] > 0,
        "temporal normalized-state clip must be positive",
    )
    _require(
        diffusion["history_frames"] == camera["history_frames"],
        "diffusion history must match the observation contract",
    )
    _require(
        diffusion["action_horizon"] == policy["action_horizon"]
        and diffusion["execute_steps"] == policy["execute_steps"],
        "diffusion action and execution horizons must match policy",
    )
    _require(
        diffusion["action_dimension"] == policy["action_dimension"],
        "diffusion action dimension must match policy",
    )
    _require(
        diffusion["state_input_dimension"] == preprocessing["model_state_dimension"]
        and diffusion["condition_input_dimension"]
        == preprocessing["categorical_condition_dimension"],
        "diffusion state and condition dimensions must match preprocessing",
    )
    _require(
        diffusion["image_encoder"] == policy["visual_encoder"]
        and diffusion["image_size"] == camera["model_width"] == camera["model_height"],
        "diffusion image contract must match the policy camera",
    )
    _require(
        diffusion["diffusion_steps"] >= diffusion["inference_steps"] > 0,
        "invalid diffusion training or inference step count",
    )
    _require(
        diffusion["denoiser_dimension"] % diffusion["denoiser_heads"] == 0,
        "diffusion denoiser dimension must be divisible by attention heads",
    )
    _require(
        0 <= diffusion["dropout"] < 1
        and 0 <= diffusion["cosine_s"] < 1
        and diffusion["ddim_eta"] >= 0,
        "invalid diffusion regularization or schedule parameters",
    )
    _require(
        diffusion["batch_size"] > 0
        and diffusion["epochs"] > 0
        and diffusion["sampling_evaluation_batches"] > 0,
        "diffusion training dimensions must be positive",
    )
    _require(
        diffusion["freeze_encoder_epochs"] < diffusion["epochs"],
        "diffusion encoder freeze period must be shorter than training",
    )
    _require(
        diffusion["noise_loss_weight"] > 0
        and diffusion["clean_action_loss_weight"] >= 0
        and diffusion["longitudinal_reconstruction_weight"] >= 1
        and diffusion["temporal_derivative_loss_weight"] >= 0,
        "diffusion objective weights are invalid",
    )
    _require(
        diffusion["selection_sampling_batches"] > 0,
        "diffusion selection sampling coverage must be positive",
    )
    _require(
        len(diffusion_evaluation["candidate_noise_seeds"]) >= 3
        and len(set(diffusion_evaluation["candidate_noise_seeds"]))
        == len(diffusion_evaluation["candidate_noise_seeds"]),
        "diffusion evaluation requires at least three unique noise seeds",
    )
    _require(
        diffusion_evaluation["maximum_relative_rmse"] >= 1,
        "diffusion relative RMSE limit cannot be below the baseline",
    )
    _require(
        diffusion_evaluation["maximum_absolute_longitudinal_bias"] >= 0
        and diffusion_evaluation["maximum_longitudinal_behavior_drift"] >= 0,
        "diffusion bias and behavior-drift limits must be non-negative",
    )
    shared_factorized_fields = (
        "history_frames",
        "action_horizon",
        "execute_steps",
        "action_dimension",
        "state_input_dimension",
        "condition_input_dimension",
        "image_encoder",
        "image_projection_dimension",
        "state_projection_dimension",
        "temporal_hidden_dimension",
        "condition_projection_dimension",
        "denoiser_dimension",
        "denoiser_layers",
        "denoiser_heads",
        "diffusion_steps",
        "inference_steps",
        "cosine_s",
        "ddim_eta",
        "image_size",
    )
    _require(
        all(factorized[field] == diffusion[field] for field in shared_factorized_fields),
        "factorized steering branch must match the released diffusion architecture",
    )
    _require(
        factorized["history_frames"] == camera["history_frames"]
        and factorized["action_horizon"] == policy["action_horizon"]
        and factorized["execute_steps"] == policy["execute_steps"],
        "factorized temporal horizons must match the deployment policy",
    )
    _require(
        factorized["longitudinal_hidden_dimension"] > 0
        and factorized["batch_size"] > 0
        and factorized["epochs"] > 0
        and factorized["selection_sampling_batches"] > 0,
        "factorized training dimensions must be positive",
    )
    _require(
        factorized["freeze_encoder_epochs"] < factorized["epochs"]
        and 0 < factorized["learning_rate"] < 1
        and factorized["weight_decay"] >= 0,
        "factorized optimizer configuration is invalid",
    )
    _require(
        0 <= factorized["longitudinal_neutral_threshold"] < 1
        and all(
            factorized["longitudinal_mode_weights"][name] > 0
            for name in ("braking", "neutral", "acceleration")
        ),
        "factorized longitudinal mode configuration is invalid",
    )
    factorized_loss_names = (
        "steering_noise_loss_weight",
        "steering_action_loss_weight",
        "steering_derivative_loss_weight",
        "longitudinal_action_loss_weight",
        "longitudinal_derivative_loss_weight",
        "longitudinal_mode_loss_weight",
    )
    _require(
        all(factorized[name] >= 0 for name in factorized_loss_names)
        and factorized["steering_noise_loss_weight"] > 0
        and factorized["longitudinal_action_loss_weight"] > 0,
        "factorized objective weights are invalid",
    )
    _require(
        len(factorized_evaluation["candidate_noise_seeds"]) >= 3
        and len(set(factorized_evaluation["candidate_noise_seeds"]))
        == len(factorized_evaluation["candidate_noise_seeds"]),
        "factorized evaluation requires at least three unique noise seeds",
    )
    _require(
        factorized_evaluation["maximum_relative_rmse"] >= 1
        and factorized_evaluation["maximum_absolute_longitudinal_bias"] >= 0
        and factorized_evaluation["maximum_longitudinal_behavior_drift"] >= 0
        and factorized_evaluation["maximum_longitudinal_smoothness_ratio"] >= 1,
        "factorized promotion thresholds are invalid",
    )
    _require(
        factorized_finetuning["batch_size"] > 0
        and factorized_finetuning["epochs"] > 0
        and 0 < factorized_finetuning["learning_rate"] < 1
        and factorized_finetuning["weight_decay"] >= 0
        and factorized_finetuning["early_stopping_patience"] > 0
        and factorized_finetuning["minimum_improvement"] >= 0
        and factorized_finetuning["gradient_clip_norm"] > 0,
        "factorized longitudinal fine-tuning optimizer is invalid",
    )
    _require(
        factorized_finetuning["first_action_mse_weight"] > 0
        and factorized_finetuning["chunk_mse_weight"] >= 0
        and factorized_finetuning["derivative_mse_weight"] >= 0,
        "factorized longitudinal fine-tuning objective is invalid",
    )
    _require(
        phase7_closed_loop["protocol_version"] == "1.0.0",
        "unsupported Phase 7 closed-loop protocol version",
    )
    _require(
        isinstance(phase7_closed_loop["checkpoint_sha256"], str)
        and len(phase7_closed_loop["checkpoint_sha256"]) == 64,
        "Phase 7 closed-loop checkpoint digest must be SHA-256",
    )
    _require(
        phase7_closed_loop["deployment_noise_seed"]
        in factorized_evaluation["candidate_noise_seeds"],
        "Phase 7 deployment noise seed must come from the offline candidates",
    )
    _require(
        phase7_closed_loop["episodes"] >= 3
        and phase7_closed_loop["ticks_per_episode"] > 0
        and phase7_closed_loop["background_vehicles"] >= 0,
        "Phase 7 closed-loop cardinality is invalid",
    )
    _require(
        0 < phase7_closed_loop["minimum_mean_route_progress_fraction"] <= 1
        and phase7_closed_loop["minimum_mean_distance_m"] > 0
        and 0 <= phase7_closed_loop["maximum_stationary_fraction"] < 1,
        "Phase 7 progress and liveness thresholds are invalid",
    )
    _require(
        all(
            phase7_closed_loop[name] >= 0
            for name in (
                "maximum_collisions",
                "maximum_lane_invasions",
                "maximum_red_light_violations",
                "maximum_no_progress_terminations",
            )
        ),
        "Phase 7 safety-event budgets must be non-negative",
    )
    _require(
        phase7_closed_loop["maximum_speed_mps"] > 0
        and phase7_closed_loop["maximum_lane_offset_m"] > 0
        and phase7_closed_loop["cold_start_ticks"] >= 1
        and phase7_closed_loop["maximum_cold_start_latency_ms"] > 0
        and 0
        < phase7_closed_loop["maximum_steady_state_latency_ms"]
        <= 1000 / simulator["control_hz"],
        "Phase 7 deployment envelope is invalid",
    )
    _require(vla["protocol_version"] == "1.0.0", "unsupported VLA protocol version")
    _require(
        vla["language_source"] == "deterministic_route_and_signal_metadata",
        "Phase 8 preflight must disclose its structured language source",
    )
    _require(
        vla["history_frames"] == camera["history_frames"]
        and vla["state_input_dimension"] == preprocessing["model_state_dimension"]
        and vla["image_size"] == camera["model_width"] == camera["model_height"],
        "VLA observation contract must match preprocessing",
    )
    _require(
        vla["action_horizon"] == policy["action_horizon"]
        and 1 <= vla["execute_steps"] <= vla["action_horizon"]
        and simulator["control_hz"] % vla["planner_hz"] == 0
        and vla["execute_steps"] == simulator["control_hz"] // vla["planner_hz"],
        "VLA hierarchy must align planner and control rates",
    )
    _require(
        vla["max_instruction_tokens"] >= 4
        and vla["language_dimension"] > 0
        and vla["state_hidden_dimension"] > 0
        and vla["fusion_dimension"] > 0,
        "VLA model dimensions are invalid",
    )
    _require(
        vla["batch_size"] > 0
        and vla["data_loader_workers"] >= 0
        and vla["epochs"] > 0
        and 0 < vla["learning_rate"] < 1
        and vla["weight_decay"] >= 0
        and vla["gradient_clip_norm"] > 0,
        "VLA training configuration is invalid",
    )
    _require(
        0 <= vla["target_maximum_safety_active_fraction"] < 1
        and 0 < vla["maximum_planner_latency_ms"] <= 1000 / vla["planner_hz"],
        "VLA promotion envelope is invalid",
    )
    _require(
        vla_evaluation["expected_validation_samples"] > 0
        and vla_evaluation["expected_test_samples"] > 0
        and vla_evaluation["maximum_relative_rmse"] >= 1
        and vla_evaluation["maximum_absolute_longitudinal_bias"] >= 0
        and vla_evaluation["maximum_longitudinal_behavior_drift"] >= 0
        and vla_evaluation["maximum_chunk_smoothness_ratio"] >= 1,
        "VLA offline promotion thresholds are invalid",
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
