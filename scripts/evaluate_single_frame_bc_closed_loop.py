#!/usr/bin/env python3
"""Evaluate the single-frame BC checkpoint under closed-loop CARLA control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import carla
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from collect_phase2_pilot import (
    build_route,
    magnitude,
    nearest_route_index,
    select_vehicle_blueprint,
    sorted_spawn_points,
    spawn_background,
    state_vector,
    update_spectator,
)

from carla_diffusion.bc_dataset import build_image_transform
from carla_diffusion.bc_model import SingleFrameBC
from carla_diffusion.closed_loop import (
    aggregate_episode_reports,
    latency_summary,
    longitudinal_to_pedals,
    route_command_from_geometry,
)
from carla_diffusion.config import load_and_validate_config
from carla_diffusion.training_data import categorical_condition, transform_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / "phase4_single_frame_bc_v062"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase3_pilot_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            PROJECT_ROOT / "artifacts" / "evaluations" / "phase4_closed_loop_report.json"
        ),
    )
    parser.add_argument("--map")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--ticks-per-episode", type=int)
    parser.add_argument("--background-vehicles", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sensor-timeout", type=float, default=30.0)
    parser.add_argument("--spectator-follow", action="store_true")
    parser.add_argument(
        "--continue-after-collision",
        action="store_true",
        help="visualization-only override; frozen evaluation still terminates on collision",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def carla_image_to_pil(image: carla.Image) -> Image.Image:
    return Image.frombuffer(
        "RGBA",
        (image.width, image.height),
        bytes(image.raw_data),
        "raw",
        "BGRA",
    ).convert("RGB")


class PolicyRuntime:
    def __init__(self, checkpoint_path: Path, processed_root: Path) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for closed-loop policy evaluation")
        self.device = torch.device("cuda")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.checkpoint_epoch = int(checkpoint["epoch"])
        self.config: dict[str, Any] = checkpoint["config"]
        bc = self.config["behavioral_cloning"]
        self.model = SingleFrameBC(
            scalar_dimension=int(bc["scalar_input_dimension"]),
            scalar_feature_dimension=int(bc["scalar_feature_dimension"]),
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
        self.normalized_state_clip = float(bc["normalized_state_clip"])
        self.image_transform = build_image_transform(int(bc["image_size"]), augment=False)
        self.warmup_iterations = 20
        self.model_type = "single_frame_bc"
        self.history_frames = 1
        dummy_image = Image.new(
            "RGB",
            (
                int(self.config["camera"]["capture_width"]),
                int(self.config["camera"]["capture_height"]),
            ),
        )
        dummy_context = torch.zeros(
            1,
            int(bc["scalar_input_dimension"]),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            for _ in range(self.warmup_iterations):
                dummy_tensor = self.image_transform(dummy_image).unsqueeze(0).to(self.device)
                self.model(dummy_tensor, dummy_context)
        torch.cuda.synchronize()

    def reset_episode(self) -> None:
        """Reset stateful policy context before a new route."""

    @torch.inference_mode()
    def predict(
        self,
        image: carla.Image,
        state: list[float],
        route_command: str,
        traffic_light_state: str,
    ) -> tuple[float, float, float]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        image_tensor = self.image_transform(carla_image_to_pil(image)).unsqueeze(0).to(self.device)
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
        condition = torch.tensor(
            categorical_condition(route_command, traffic_light_state),
            dtype=torch.float32,
            device=self.device,
        )
        context = torch.cat((normalized_state, condition)).unsqueeze(0)
        prediction = self.model(image_tensor, context)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not bool(torch.all(torch.isfinite(prediction))):
            raise FloatingPointError("policy produced a non-finite closed-loop action")
        steering, longitudinal = prediction[0].detach().cpu().tolist()
        return float(steering), float(longitudinal), latency_ms


def route_command(
    route: list[carla.Waypoint],
    route_index: int,
    lookahead: int,
    turn_threshold: float,
) -> str:
    future_index = min(len(route) - 1, route_index + lookahead)
    junction_ahead = any(
        waypoint.is_junction for waypoint in route[route_index : future_index + 1]
    )
    return route_command_from_geometry(
        route[route_index].transform.rotation.yaw,
        route[future_index].transform.rotation.yaw,
        junction_ahead=junction_ahead,
        turn_threshold_degrees=turn_threshold,
    )


def evaluate_episode(
    *,
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    policy: PolicyRuntime,
    config: dict[str, Any],
    episode_number: int,
    seed: int,
    ticks: int,
    background_vehicles: int,
    traffic_manager_port: int,
    sensor_timeout: float,
    spectator_follow: bool,
) -> dict[str, Any]:
    closed_loop = config["closed_loop_evaluation"]
    episode_seed = seed + episode_number
    spawn_points = sorted_spawn_points(world)
    spawn_index = episode_seed % len(spawn_points)
    route = build_route(
        world.get_map(),
        spawn_points[spawn_index],
        episode_seed,
        float(closed_loop["route_spacing_m"]),
        int(closed_loop["route_points"]),
    )
    actors: list[carla.Actor] = []
    images: queue.Queue[carla.Image] = queue.Queue(maxsize=32)
    collision_frames: set[int] = set()
    collision_details: list[dict[str, Any]] = []
    lane_invasion_frames: set[int] = set()
    route_index = 0
    latencies: list[float] = []
    speeds: list[float] = []
    steering_rates: list[float] = []
    longitudinal_rates: list[float] = []
    abs_lane_offsets: list[float] = []
    command_counts: Counter[str] = Counter()
    distance_traveled = 0.0
    red_armed = False
    red_light_violations = 0
    previous_location: carla.Location | None = None
    previous_steering: float | None = None
    previous_longitudinal: float | None = None
    terminated_reason = "tick_limit"
    ticks_completed = 0
    delta = float(config["simulator"]["fixed_delta_seconds"])
    policy.reset_episode()

    try:
        vehicle = world.try_spawn_actor(
            select_vehicle_blueprint(world, "hero"), spawn_points[spawn_index]
        )
        if vehicle is None:
            raise RuntimeError(f"failed to spawn policy vehicle at index {spawn_index}")
        actors.append(vehicle)

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(config["camera"]["capture_width"]))
        camera_bp.set_attribute("image_size_y", str(config["camera"]["capture_height"]))
        camera_bp.set_attribute("fov", str(config["camera"]["fov_degrees"]))
        camera_bp.set_attribute("sensor_tick", "0.0")
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=2.2)),
            attach_to=vehicle,
        )
        camera.listen(images.put)
        actors.append(camera)

        collision = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.collision"),
            carla.Transform(),
            attach_to=vehicle,
        )
        def record_collision(event: carla.CollisionEvent) -> None:
            collision_frames.add(event.frame)
            collision_details.append(
                {
                    "frame": event.frame,
                    "other_actor_type": event.other_actor.type_id,
                    "normal_impulse": magnitude(event.normal_impulse),
                }
            )

        collision.listen(record_collision)
        actors.append(collision)
        invasion = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.lane_invasion"),
            carla.Transform(),
            attach_to=vehicle,
        )
        invasion.listen(lambda event: lane_invasion_frames.add(event.frame))
        actors.append(invasion)

        background = spawn_background(
            world,
            traffic_manager_port,
            spawn_points,
            spawn_index,
            background_vehicles,
            episode_seed,
        )
        actors.extend(background)

        for _ in range(ticks):
            expected_frame = world.tick()
            if spectator_follow:
                update_spectator(world, vehicle)
            try:
                image = images.get(timeout=sensor_timeout)
            except queue.Empty as error:
                raise RuntimeError(f"camera timeout at frame {expected_frame}") from error
            if image.frame != expected_frame:
                raise RuntimeError(
                    f"camera/world frame mismatch: camera={image.frame}, world={expected_frame}"
                )
            state, light_state, in_junction = state_vector(vehicle, world)
            current_transform = vehicle.get_transform()
            route_index = nearest_route_index(current_transform, route, route_index)
            command = route_command(
                route,
                route_index,
                int(closed_loop["command_lookahead_points"]),
                float(closed_loop["turn_threshold_degrees"]),
            )
            steering, longitudinal, latency_ms = policy.predict(
                image, state, command, light_state
            )
            throttle, brake = longitudinal_to_pedals(longitudinal)
            vehicle.apply_control(
                carla.VehicleControl(
                    throttle=throttle,
                    brake=brake,
                    steer=max(-1.0, min(1.0, steering)),
                )
            )

            location = current_transform.location
            if previous_location is not None:
                distance_traveled += location.distance(previous_location)
            if previous_steering is not None and previous_longitudinal is not None:
                steering_rates.append(abs(steering - previous_steering) / delta)
                longitudinal_rates.append(abs(longitudinal - previous_longitudinal) / delta)
            previous_location = location
            previous_steering = steering
            previous_longitudinal = longitudinal
            speeds.append(float(state[0]))
            abs_lane_offsets.append(abs(float(state[3])))
            latencies.append(latency_ms)
            command_counts[command] += 1
            ticks_completed += 1

            if light_state == "red" and not in_junction:
                red_armed = True
            red_violation = red_armed and in_junction and state[0] > 1.0
            if red_violation:
                red_light_violations += 1
                red_armed = False
            elif light_state in {"green", "yellow", "none"} and not in_junction:
                red_armed = False

            if route_index >= len(route) - 3:
                terminated_reason = "route_complete"
                break
            if collision_frames and bool(closed_loop["terminate_on_collision"]):
                terminated_reason = "collision"
                break

        latency = latency_summary(latencies)
        return {
            "episode": episode_number,
            "seed": episode_seed,
            "spawn_index": spawn_index,
            "ticks_completed": ticks_completed,
            "terminated_reason": terminated_reason,
            "route_points": len(route),
            "route_index": route_index,
            "route_progress_fraction": route_index / max(1, len(route) - 1),
            "distance_traveled_m": distance_traveled,
            "collision_events": len(collision_details),
            "collision_details": collision_details,
            "lane_invasion_events": len(lane_invasion_frames),
            "red_light_violations": red_light_violations,
            "mean_speed_mps": sum(speeds) / len(speeds),
            "max_speed_mps": max(speeds),
            "stationary_fraction": sum(speed < 0.5 for speed in speeds) / len(speeds),
            "mean_abs_lane_offset_m": sum(abs_lane_offsets) / len(abs_lane_offsets),
            "max_abs_lane_offset_m": max(abs_lane_offsets),
            "mean_abs_steering_rate_per_second": (
                sum(steering_rates) / len(steering_rates) if steering_rates else 0.0
            ),
            "mean_abs_longitudinal_rate_per_second": (
                sum(longitudinal_rates) / len(longitudinal_rates)
                if longitudinal_rates
                else 0.0
            ),
            "route_command_counts": dict(sorted(command_counts.items())),
            "policy_pipeline_latency": latency,
        }
    finally:
        for actor in reversed(actors):
            try:
                if "sensor" in actor.type_id:
                    actor.stop()
                actor.destroy()
            except RuntimeError:
                pass
        try:
            world.tick()
        except RuntimeError:
            pass


def main() -> int:
    args = parse_args()
    project_config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    closed_loop = project_config["closed_loop_evaluation"]
    if args.continue_after_collision:
        closed_loop["terminate_on_collision"] = False
    checkpoint_path = args.checkpoint.resolve()
    processed_root = args.processed_root.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    map_name = str(args.map or closed_loop["map"])
    episodes = int(args.episodes or closed_loop["episodes"])
    ticks = int(args.ticks_per_episode or closed_loop["ticks_per_episode"])
    background_vehicles = int(
        args.background_vehicles
        if args.background_vehicles is not None
        else closed_loop["background_vehicles"]
    )
    seed = int(args.seed or closed_loop["seed"])
    if episodes < 1 or ticks < 1 or background_vehicles < 0:
        raise ValueError("invalid closed-loop evaluation cardinality")

    policy = PolicyRuntime(checkpoint_path, processed_root)
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.get_world()
    current_map = world.get_map().name.rsplit("/", 1)[-1]
    if current_map != map_name:
        world = client.load_world(map_name)
    else:
        print(f"Reusing already-loaded map {current_map}", flush=True)
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    started = time.perf_counter()
    episode_reports: list[dict[str, Any]] = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = project_config["simulator"]["fixed_delta_seconds"]
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(seed)
        for episode_number in range(episodes):
            print(f"Evaluating closed-loop episode {episode_number + 1}/{episodes}", flush=True)
            report = evaluate_episode(
                world=world,
                traffic_manager=traffic_manager,
                policy=policy,
                config=project_config,
                episode_number=episode_number,
                seed=seed,
                ticks=ticks,
                background_vehicles=background_vehicles,
                traffic_manager_port=args.traffic_manager_port,
                sensor_timeout=args.sensor_timeout,
                spectator_follow=args.spectator_follow,
            )
            episode_reports.append(report)
            print(json.dumps(report), flush=True)
    finally:
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        world.apply_settings(original_settings)

    aggregate = aggregate_episode_reports(episode_reports)
    all_p95 = [
        float(episode["policy_pipeline_latency"]["p95_ms"])
        for episode in episode_reports
    ]
    aggregate["max_episode_p95_policy_pipeline_latency_ms"] = max(all_p95)
    all_maximums = [
        float(episode["policy_pipeline_latency"]["max_ms"])
        for episode in episode_reports
    ]
    aggregate["max_policy_pipeline_latency_ms"] = max(all_maximums)
    aggregate["latency_budget_ms"] = project_config["evaluation"]["max_policy_latency_ms"]
    aggregate["latency_gate_passed"] = max(all_maximums) <= float(
        project_config["evaluation"]["max_policy_latency_ms"]
    )
    final_report = {
        "status": "passed",
        "model_type": policy.model_type,
        "observation_history_frames": policy.history_frames,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "checkpoint_epoch": policy.checkpoint_epoch,
        "policy_warmup_iterations": policy.warmup_iterations,
        "terminate_on_collision": bool(closed_loop["terminate_on_collision"]),
        "map": world.get_map().name,
        "seed": seed,
        "episodes": episode_reports,
        "aggregate": aggregate,
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_json(args.report.resolve(), final_report)
    print(json.dumps(final_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
