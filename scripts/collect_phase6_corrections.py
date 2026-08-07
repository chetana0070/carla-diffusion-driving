#!/usr/bin/env python3
"""Collect expert recovery segments triggered by temporal-BC failure precursors."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import carla

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from collect_phase2_pilot import (
    build_route,
    encode_jpeg,
    nearest_route_index,
    select_vehicle_blueprint,
    sorted_spawn_points,
    spawn_background,
    state_vector,
    update_spectator,
)
from evaluate_single_frame_bc_closed_loop import route_command
from evaluate_temporal_bc_closed_loop import TemporalPolicyRuntime

from carla_diffusion.closed_loop import longitudinal_to_pedals
from carla_diffusion.config import load_and_validate_config
from carla_diffusion.dataset import AtomicEpisodeWriter, DatasetError, free_gib, validate_dataset
from carla_diffusion.intervention import InterventionMonitor, InterventionThresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--map")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--ticks-per-episode", type=int)
    parser.add_argument("--background-vehicles", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sensor-timeout", type=float, default=30.0)
    parser.add_argument("--spectator-follow", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "checkpoints"
            / "phase5_temporal_bc_v080"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "phase3_pilot_v1",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "phase6_corrections_v1",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluations" / "phase6_collection_report.json",
    )
    return parser.parse_args()


def thresholds(config: dict[str, Any]) -> InterventionThresholds:
    corrective = config["corrective_collection"]
    return InterventionThresholds(
        warmup_ticks=int(corrective["warmup_ticks"]),
        lane_offset_m=float(corrective["lane_offset_trigger_m"]),
        heading_error_degrees=float(corrective["heading_error_trigger_degrees"]),
        unsafe_lead_distance_m=float(corrective["unsafe_lead_distance_m"]),
        unsafe_lead_min_longitudinal=float(corrective["unsafe_lead_min_longitudinal"]),
        stall_ticks=int(corrective["stall_trigger_ticks"]),
        stationary_speed_mps=float(corrective["stationary_speed_mps"]),
    )


def collect_episode(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    policy: TemporalPolicyRuntime,
    episode_number: int,
    ticks: int,
    background_vehicles: int,
    seed: int,
) -> dict[str, Any]:
    corrective = config["corrective_collection"]
    closed_loop = config["closed_loop_evaluation"]
    episode_seed = seed + episode_number
    episode_id = f"town01-correction-{episode_number:03d}-seed-{episode_seed}"
    route_id = f"town01-correction-route-{episode_number:03d}-seed-{episode_seed}"
    spawn_points = sorted_spawn_points(world)
    spawn_index = episode_seed % len(spawn_points)
    route = build_route(
        world.get_map(),
        spawn_points[spawn_index],
        episode_seed,
        float(closed_loop["route_spacing_m"]),
        int(closed_loop["route_points"]),
    )
    monitor = InterventionMonitor(thresholds(config))
    actors: list[carla.Actor] = []
    images: queue.Queue[carla.Image] = queue.Queue(maxsize=32)
    collision_frames: set[int] = set()
    lane_invasion_frames: set[int] = set()
    writer: AtomicEpisodeWriter | None = None
    intervention_reasons: tuple[str, ...] = ()
    intervention_tick: int | None = None
    correction_samples = 0
    route_index = 0
    policy_ticks = 0
    outcome = "no_intervention"
    first_correction_frame: int | None = None
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
        collision.listen(lambda event: collision_frames.add(event.frame))
        actors.append(collision)
        invasion = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.lane_invasion"),
            carla.Transform(),
            attach_to=vehicle,
        )
        invasion.listen(lambda event: lane_invasion_frames.add(event.frame))
        actors.append(invasion)
        actors.extend(
            spawn_background(
                world,
                args.traffic_manager_port,
                spawn_points,
                spawn_index,
                background_vehicles,
                episode_seed,
            )
        )

        for tick_index in range(ticks):
            if tick_index % 100 == 0 and free_gib(args.dataset_root) <= float(
                corrective["stop_collection_free_disk_gib"]
            ):
                raise DatasetError("corrective collection reached the disk safety threshold")
            expected_frame = world.tick()
            if args.spectator_follow:
                update_spectator(world, vehicle)
            try:
                image = images.get(timeout=args.sensor_timeout)
            except queue.Empty as error:
                raise RuntimeError(f"camera timeout at frame {expected_frame}") from error
            if image.frame != expected_frame:
                raise RuntimeError(
                    f"camera/world frame mismatch: camera={image.frame}, world={expected_frame}"
                )
            state, light_state, _ = state_vector(vehicle, world)
            route_index = nearest_route_index(vehicle.get_transform(), route, route_index)
            command = route_command(
                route,
                route_index,
                int(closed_loop["command_lookahead_points"]),
                float(closed_loop["turn_threshold_degrees"]),
            )

            if intervention_tick is None:
                steering, longitudinal, _ = policy.predict(
                    image, state, command, light_state
                )
                intervention_reasons = monitor.update(state, light_state, longitudinal)
                policy_ticks += 1
                if intervention_reasons:
                    intervention_tick = tick_index
                    vehicle.set_autopilot(True, args.traffic_manager_port)
                    traffic_manager.set_path(
                        vehicle,
                        [waypoint.transform.location for waypoint in route[route_index + 1 :]],
                    )
                    metadata = {
                        "schema_version": "1.0.0",
                        "collector_version": "1.0.0",
                        "collector_type": "intervention_expert_recovery",
                        "carla_server_version": "0.9.16",
                        "map": world.get_map().name,
                        "seed": episode_seed,
                        "spawn_index": spawn_index,
                        "requested_ticks": ticks,
                        "fixed_delta_seconds": config["simulator"]["fixed_delta_seconds"],
                        "rollout_policy": "temporal_bc",
                        "expert": "carla_traffic_manager",
                        "intervention_tick": intervention_tick,
                        "intervention_reasons": list(intervention_reasons),
                    }
                    writer = AtomicEpisodeWriter(
                        args.dataset_root, episode_id, route_id, metadata
                    )
                    outcome = "expert_recovery"
                    continue
                throttle, brake = longitudinal_to_pedals(longitudinal)
                vehicle.apply_control(
                    carla.VehicleControl(
                        steer=max(-1.0, min(1.0, steering)),
                        throttle=throttle,
                        brake=brake,
                    )
                )
            else:
                control = vehicle.get_control()
                longitudinal = max(
                    -1.0,
                    min(1.0, float(control.throttle) - float(control.brake)),
                )
                first_correction_frame = (
                    image.frame if first_correction_frame is None else first_correction_frame
                )
                image_relative = f"{episode_id}/images/{image.frame:08d}.jpg"
                record = {
                    "schema_version": "1.0.0",
                    "episode_id": episode_id,
                    "route_id": route_id,
                    "frame_id": image.frame,
                    "timestamp_seconds": (
                        image.frame - first_correction_frame
                    )
                    * float(config["simulator"]["fixed_delta_seconds"]),
                    "observation": {
                        "rgb_front_path": image_relative,
                        "state": state,
                        "traffic_light_state": light_state,
                        "route_command": command,
                    },
                    "expert_action": [float(control.steer), longitudinal],
                    "events": {
                        "collision": image.frame in collision_frames,
                        "lane_invasion": image.frame in lane_invasion_frames,
                        "red_light_violation": False,
                        "intervention": True,
                    },
                }
                if writer is None:
                    raise RuntimeError("intervention writer was not initialized")
                writer.add(record, encode_jpeg(image, int(config["camera"]["jpeg_quality"])))
                correction_samples += 1
                if correction_samples >= int(corrective["expert_recovery_ticks"]):
                    break
            if collision_frames:
                outcome = "collision_before_intervention"
                break

        published = False
        if writer is not None:
            if correction_samples >= int(corrective["minimum_published_samples"]):
                writer.finalize(
                    {
                        "policy_ticks": policy_ticks,
                        "correction_samples": correction_samples,
                        "route_progress_fraction": route_index / max(1, len(route) - 1),
                        "collision_events": len(collision_frames),
                        "lane_invasion_events": len(lane_invasion_frames),
                    }
                )
                published = True
            else:
                writer.abort("insufficient expert recovery samples")
        return {
            "episode": episode_number,
            "seed": episode_seed,
            "outcome": outcome,
            "published": published,
            "policy_ticks": policy_ticks,
            "correction_samples": correction_samples,
            "intervention_tick": intervention_tick,
            "intervention_reasons": list(intervention_reasons),
            "collision_events": len(collision_frames),
            "lane_invasion_events": len(lane_invasion_frames),
        }
    except Exception as error:
        if writer is not None:
            writer.abort(f"{type(error).__name__}: {error}")
        raise
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
    config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    corrective = config["corrective_collection"]
    episodes = int(args.episodes or corrective["episodes"])
    ticks = int(args.ticks_per_episode or corrective["ticks_per_episode"])
    background = int(
        args.background_vehicles
        if args.background_vehicles is not None
        else corrective["background_vehicles"]
    )
    seed = int(args.seed or corrective["seed"])
    map_name = str(args.map or config["closed_loop_evaluation"]["map"])
    if episodes < 1 or ticks < 1 or background < 0:
        raise ValueError("invalid corrective collection cardinality")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    if any(path.is_dir() for path in args.dataset_root.iterdir()):
        raise DatasetError(f"corrective dataset root is not empty: {args.dataset_root}")
    available = free_gib(args.dataset_root)
    if available < float(corrective["minimum_free_disk_gib"]):
        raise DatasetError(f"insufficient free disk for collection: {available:.1f} GiB")

    policy = TemporalPolicyRuntime(checkpoint, args.processed_root.resolve())
    client = carla.Client(args.host, args.port)
    client.set_timeout(120.0)
    world = client.get_world()
    if world.get_map().name.rsplit("/", 1)[-1] != map_name:
        world = client.load_world(map_name)
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config["simulator"]["fixed_delta_seconds"]
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(seed)
        for episode_number in range(episodes):
            print(f"Collecting corrective rollout {episode_number + 1}/{episodes}", flush=True)
            report = collect_episode(
                args=args,
                config=config,
                world=world,
                traffic_manager=traffic_manager,
                policy=policy,
                episode_number=episode_number,
                ticks=ticks,
                background_vehicles=background,
                seed=seed,
            )
            reports.append(report)
            print(json.dumps(report), flush=True)
    finally:
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        world.apply_settings(original_settings)

    validation = validate_dataset(args.dataset_root)
    published = sum(bool(report["published"]) for report in reports)
    reasons = Counter(
        reason for report in reports for reason in report["intervention_reasons"]
    )
    status = "passed" if published > 0 and validation["status"] == "passed" else "failed"
    final = {
        "status": status,
        "requested_episodes": episodes,
        "published_intervention_episodes": published,
        "total_correction_samples": sum(
            int(report["correction_samples"]) for report in reports
        ),
        "intervention_reason_counts": dict(sorted(reasons.items())),
        "episodes": reports,
        "validation": validation,
        "free_disk_gib_after": round(free_gib(args.dataset_root), 3),
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
