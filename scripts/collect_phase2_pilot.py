#!/usr/bin/env python3
"""Collect deterministic Traffic Manager expert episodes for the Phase 2 pilot."""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import carla
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carla_diffusion.config import load_and_validate_config
from carla_diffusion.dataset import AtomicEpisodeWriter, DatasetError, free_gib, validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--map", default="Town01")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--ticks-per-episode", type=int, default=600)
    parser.add_argument("--background-vehicles", type=int, default=8)
    parser.add_argument("--guaranteed-lead-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--route-spacing", type=float, default=5.0)
    parser.add_argument("--route-points", type=int, default=180)
    parser.add_argument("--sensor-timeout", type=float, default=30.0)
    parser.add_argument("--spectator-follow", action="store_true")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "phase2_pilot",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "evaluations" / "phase2_pilot_report.json",
    )
    return parser.parse_args()


def magnitude(vector: carla.Vector3D) -> float:
    return math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)


def dot(vector: carla.Vector3D, direction: carla.Vector3D) -> float:
    return vector.x * direction.x + vector.y * direction.y + vector.z * direction.z


def angle_difference_radians(target_degrees: float, source_degrees: float) -> float:
    difference = (target_degrees - source_degrees + 180.0) % 360.0 - 180.0
    return math.radians(difference)


def sorted_spawn_points(world: carla.World) -> list[carla.Transform]:
    return sorted(
        world.get_map().get_spawn_points(),
        key=lambda transform: (
            round(transform.location.x, 3),
            round(transform.location.y, 3),
            round(transform.location.z, 3),
        ),
    )


def select_vehicle_blueprint(world: carla.World, role_name: str) -> carla.ActorBlueprint:
    library = world.get_blueprint_library()
    preferred = library.filter("vehicle.tesla.model3")
    candidates = preferred or library.filter("vehicle.*")
    if not candidates:
        raise RuntimeError("no vehicle blueprint is available")
    blueprint = min(candidates, key=lambda item: item.id)
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", role_name)
    return blueprint


def build_route(
    world_map: carla.Map,
    start: carla.Transform,
    seed: int,
    spacing: float,
    points: int,
) -> list[carla.Waypoint]:
    rng = random.Random(seed)
    waypoint = world_map.get_waypoint(start.location, project_to_road=True)
    route = [waypoint]
    for _ in range(points - 1):
        candidates = waypoint.next(spacing)
        if not candidates:
            break
        candidates = sorted(
            candidates,
            key=lambda item: (
                round(item.transform.location.x, 3),
                round(item.transform.location.y, 3),
                round(item.transform.rotation.yaw, 3),
            ),
        )
        forward_candidates = [
            item
            for item in candidates
            if abs(
                math.degrees(
                    angle_difference_radians(
                        item.transform.rotation.yaw,
                        waypoint.transform.rotation.yaw,
                    )
                )
            )
            < 120.0
        ]
        waypoint = rng.choice(forward_candidates or candidates)
        route.append(waypoint)
    if len(route) < 20:
        raise RuntimeError("generated route is too short")
    return route


def nearest_route_index(
    vehicle_transform: carla.Transform,
    route: list[carla.Waypoint],
    previous_index: int,
) -> int:
    upper = min(len(route), previous_index + 15)
    return min(
        range(previous_index, upper),
        key=lambda item: route[item].transform.location.distance(vehicle_transform.location),
    )


def expert_route_command(
    traffic_manager: carla.TrafficManager,
    vehicle: carla.Vehicle,
) -> str:
    """Read the command from the same Traffic Manager that controls the expert."""
    action = traffic_manager.get_next_action(vehicle)
    option = str(action[0]).lower() if action else ""
    if "left" in option:
        return "left"
    if "right" in option:
        return "right"
    if "straight" in option:
        return "straight"
    return "follow_lane"


def traffic_light_observation(
    vehicle: carla.Vehicle,
    world: carla.World,
    waypoint: carla.Waypoint,
) -> tuple[str, float]:
    mapping = {
        carla.TrafficLightState.Red: "red",
        carla.TrafficLightState.Yellow: "yellow",
        carla.TrafficLightState.Green: "green",
        carla.TrafficLightState.Off: "unknown",
        carla.TrafficLightState.Unknown: "unknown",
    }
    vehicle_location = vehicle.get_location()
    forward = vehicle.get_transform().get_forward_vector()
    candidates: list[tuple[float, carla.TrafficLight]] = []
    for light in world.get_traffic_lights_from_waypoint(waypoint, 80.0):
        for stop_waypoint in light.get_stop_waypoints():
            same_lane = (
                stop_waypoint.road_id == waypoint.road_id
                and stop_waypoint.lane_id == waypoint.lane_id
            )
            if not same_lane:
                continue
            offset = stop_waypoint.transform.location - vehicle_location
            if dot(offset, forward) < -1.0:
                continue
            candidates.append((magnitude(offset), light))
    if not candidates:
        return "none", -1.0
    distance, light = min(candidates, key=lambda item: item[0])
    return mapping.get(light.get_state(), "unknown"), float(distance)


LEAD_VEHICLE_MAX_DISTANCE_M = 80.0


def lead_vehicle_observation(vehicle: carla.Vehicle, world: carla.World) -> tuple[float, float]:
    """Return the nearest vehicle ahead in the ego lane, within sensor range."""
    transform = vehicle.get_transform()
    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    ego_velocity = vehicle.get_velocity()
    world_map = world.get_map()
    ego_waypoint = world_map.get_waypoint(transform.location, project_to_road=True)
    best_distance = math.inf
    relative_speed = 0.0
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == vehicle.id:
            continue
        actor_waypoint = world_map.get_waypoint(actor.get_location(), project_to_road=True)
        if (
            actor_waypoint.road_id != ego_waypoint.road_id
            or actor_waypoint.lane_id != ego_waypoint.lane_id
        ):
            continue
        offset = actor.get_location() - transform.location
        longitudinal = dot(offset, forward)
        lateral = abs(dot(offset, right))
        if (
            0.0 < longitudinal <= LEAD_VEHICLE_MAX_DISTANCE_M
            and longitudinal < best_distance
            and lateral < 2.5
        ):
            best_distance = longitudinal
            relative_speed = dot(actor.get_velocity(), forward) - dot(ego_velocity, forward)
    if math.isinf(best_distance):
        return -1.0, 0.0
    return float(best_distance), float(relative_speed)


def state_vector(vehicle: carla.Vehicle, world: carla.World) -> tuple[list[float], str, bool]:
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    forward = transform.get_forward_vector()
    waypoint = world.get_map().get_waypoint(transform.location, project_to_road=True)
    lane_right = waypoint.transform.get_right_vector()
    lane_delta = transform.location - waypoint.transform.location
    lane_offset = dot(lane_delta, lane_right)
    heading_error = angle_difference_radians(
        waypoint.transform.rotation.yaw,
        transform.rotation.yaw,
    )
    lead_distance, lead_relative_speed = lead_vehicle_observation(vehicle, world)
    light_state, light_distance = traffic_light_observation(vehicle, world, waypoint)
    return (
        [
            magnitude(velocity),
            dot(acceleration, forward),
            float(vehicle.get_speed_limit()) / 3.6,
            lane_offset,
            heading_error,
            lead_distance,
            lead_relative_speed,
            light_distance,
        ],
        light_state,
        waypoint.is_junction,
    )


def encode_jpeg(image: carla.Image, quality: int) -> bytes:
    rgb = Image.frombuffer(
        "RGBA",
        (image.width, image.height),
        bytes(image.raw_data),
        "raw",
        "BGRA",
    ).convert("RGB")
    buffer = BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality, optimize=False)
    return buffer.getvalue()


def update_spectator(world: carla.World, vehicle: carla.Vehicle) -> None:
    transform = vehicle.get_transform()
    forward = transform.get_forward_vector()
    location = carla.Location(
        x=transform.location.x - 8.0 * forward.x,
        y=transform.location.y - 8.0 * forward.y,
        z=transform.location.z + 5.0,
    )
    rotation = carla.Rotation(pitch=-18.0, yaw=transform.rotation.yaw)
    world.get_spectator().set_transform(carla.Transform(location, rotation))


def spawn_background(
    world: carla.World,
    traffic_manager_port: int,
    spawn_points: list[carla.Transform],
    excluded_index: int,
    count: int,
    seed: int,
) -> list[carla.Vehicle]:
    rng = random.Random(seed)
    indices = [index for index in range(len(spawn_points)) if index != excluded_index]
    rng.shuffle(indices)
    actors: list[carla.Vehicle] = []
    for index in indices:
        if len(actors) >= count:
            break
        blueprint = select_vehicle_blueprint(world, "background")
        actor = world.try_spawn_actor(blueprint, spawn_points[index])
        if actor is not None:
            actor.set_autopilot(True, traffic_manager_port)
            actors.append(actor)
    return actors


def spawn_route_lead(
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    traffic_manager_port: int,
    route: list[carla.Waypoint],
) -> tuple[carla.Vehicle, int]:
    """Spawn a slower lead vehicle 25--50 m ahead on the expert route."""
    upper = min(11, len(route) - 1)
    for route_index in range(5, upper):
        route_transform = route[route_index].transform
        spawn_transform = carla.Transform(
            carla.Location(
                x=route_transform.location.x,
                y=route_transform.location.y,
                z=route_transform.location.z + 0.5,
            ),
            route_transform.rotation,
        )
        lead = world.try_spawn_actor(
            select_vehicle_blueprint(world, "lead"),
            spawn_transform,
        )
        if lead is None:
            continue
        lead.set_autopilot(True, traffic_manager_port)
        traffic_manager.set_path(
            lead,
            [waypoint.transform.location for waypoint in route[route_index + 1 :]],
        )
        traffic_manager.vehicle_percentage_speed_difference(lead, 30.0)
        return lead, route_index
    raise RuntimeError("failed to spawn a guaranteed lead vehicle on the route")


def collect_episode(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    episode_number: int,
    stop_free_gib: float,
) -> dict[str, Any]:
    episode_seed = args.seed + episode_number
    episode_id = f"town01-pilot-{episode_number:03d}-seed-{episode_seed}"
    route_id = f"town01-route-{episode_number:03d}-seed-{episode_seed}"
    spawn_points = sorted_spawn_points(world)
    spawn_index = (args.seed + episode_number * 17) % len(spawn_points)
    route = build_route(
        world.get_map(),
        spawn_points[spawn_index],
        episode_seed,
        args.route_spacing,
        args.route_points,
    )
    metadata = {
        "schema_version": "1.0.0",
        "collector_version": "0.4.3",
        "carla_server_version": "0.9.16",
        "map": world.get_map().name,
        "seed": episode_seed,
        "spawn_index": spawn_index,
        "requested_ticks": args.ticks_per_episode,
        "fixed_delta_seconds": config["simulator"]["fixed_delta_seconds"],
        "expert": "carla_traffic_manager",
        "lead_vehicle_max_distance_m": LEAD_VEHICLE_MAX_DISTANCE_M,
    }
    writer = AtomicEpisodeWriter(args.dataset_root, episode_id, route_id, metadata)
    actors: list[carla.Actor] = []
    images: queue.Queue[carla.Image] = queue.Queue(maxsize=32)
    collision_frames: set[int] = set()
    lane_invasion_frames: set[int] = set()
    route_index = 0
    red_armed = False
    red_light_violations = 0
    guaranteed_lead = episode_number < args.guaranteed_lead_episodes
    lead_route_index: int | None = None

    try:
        vehicle = world.try_spawn_actor(
            select_vehicle_blueprint(world, "hero"),
            spawn_points[spawn_index],
        )
        if vehicle is None:
            raise RuntimeError(f"failed to spawn ego vehicle at index {spawn_index}")
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

        collision_bp = world.get_blueprint_library().find("sensor.other.collision")
        collision = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        collision.listen(lambda event: collision_frames.add(event.frame))
        actors.append(collision)

        invasion_bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        invasion = world.spawn_actor(invasion_bp, carla.Transform(), attach_to=vehicle)
        invasion.listen(lambda event: lane_invasion_frames.add(event.frame))
        actors.append(invasion)

        if guaranteed_lead:
            lead, lead_route_index = spawn_route_lead(
                world,
                traffic_manager,
                args.traffic_manager_port,
                route,
            )
            actors.append(lead)

        background = spawn_background(
            world,
            args.traffic_manager_port,
            spawn_points,
            spawn_index,
            args.background_vehicles,
            episode_seed,
        )
        actors.extend(background)
        vehicle.set_autopilot(True, args.traffic_manager_port)
        traffic_manager.set_path(vehicle, [waypoint.transform.location for waypoint in route[1:]])

        first_frame: int | None = None
        for tick_index in range(args.ticks_per_episode):
            if tick_index % 100 == 0 and free_gib(args.dataset_root) <= stop_free_gib:
                raise DatasetError(
                    f"collection stopped at disk safety threshold ({stop_free_gib:.1f} GiB free)"
                )
            expected_frame = world.tick()
            if args.spectator_follow:
                update_spectator(world, vehicle)
            try:
                image = images.get(timeout=args.sensor_timeout)
            except queue.Empty as error:
                raise RuntimeError(
                    f"camera produced no image for frame {expected_frame} "
                    f"within {args.sensor_timeout:.1f}s"
                ) from error
            if image.frame != expected_frame:
                raise RuntimeError(
                    f"camera/world frame mismatch: camera={image.frame}, world={expected_frame}"
                )
            first_frame = image.frame if first_frame is None else first_frame

            transform = vehicle.get_transform()
            route_index = nearest_route_index(transform, route, route_index)
            command = expert_route_command(traffic_manager, vehicle)
            state, light_state, in_junction = state_vector(vehicle, world)
            if light_state == "red" and not in_junction:
                red_armed = True
            red_violation = red_armed and in_junction and state[0] > 1.0
            if red_violation:
                red_light_violations += 1
                red_armed = False
            elif light_state in {"green", "yellow", "none"} and not in_junction:
                red_armed = False

            control = vehicle.get_control()
            longitudinal = max(-1.0, min(1.0, float(control.throttle) - float(control.brake)))
            image_relative = f"{episode_id}/images/{image.frame:08d}.jpg"
            record = {
                "schema_version": "1.0.0",
                "episode_id": episode_id,
                "route_id": route_id,
                "frame_id": image.frame,
                "timestamp_seconds": (image.frame - first_frame)
                * config["simulator"]["fixed_delta_seconds"],
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
                    "red_light_violation": red_violation,
                    "intervention": False,
                },
            }
            writer.add(record, encode_jpeg(image, config["camera"]["jpeg_quality"]))

        summary = writer.finalize(
            {
                "route_points": len(route),
                "route_progress_fraction": route_index / max(1, len(route) - 1),
                "collision_events": len(collision_frames),
                "lane_invasion_events": len(lane_invasion_frames),
                "red_light_violations": red_light_violations,
                "guaranteed_lead_vehicle": guaranteed_lead,
                "lead_spawn_route_index": lead_route_index,
            }
        )
        return {
            "episode_id": summary.episode_id,
            "route_id": summary.route_id,
            "samples": summary.samples,
            "image_bytes": summary.image_bytes,
            "route_progress_fraction": route_index / max(1, len(route) - 1),
            "guaranteed_lead_vehicle": guaranteed_lead,
        }
    except Exception as error:
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
    if args.episodes < 1 or args.ticks_per_episode < 1:
        raise ValueError("episodes and ticks-per-episode must be positive")
    if not 0 <= args.guaranteed_lead_episodes <= args.episodes:
        raise ValueError("guaranteed-lead-episodes must be between 0 and episodes")
    config = load_and_validate_config(PROJECT_ROOT / "configs" / "project.json")
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    minimum_free = float(config["dataset"]["minimum_free_disk_gib"])
    stop_free = float(config["dataset"]["stop_collection_free_disk_gib"])
    available = free_gib(args.dataset_root)
    expected_samples = args.episodes * args.ticks_per_episode
    completed_directories = [
        path
        for path in args.dataset_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    if completed_directories:
        existing_validation = validate_dataset(args.dataset_root)
        correct_cardinality = (
            existing_validation.get("episodes") == args.episodes
            and existing_validation.get("samples") == expected_samples
        )
        if existing_validation["status"] == "passed" and correct_cardinality:
            report = {
                "status": "passed",
                "failure": None,
                "reused_existing_dataset": True,
                "requested_episodes": args.episodes,
                "completed_episodes": existing_validation["episodes"],
                "ticks_per_episode": args.ticks_per_episode,
                "elapsed_wall_seconds": 0.0,
                "free_disk_gib_after": round(free_gib(args.dataset_root), 3),
                "episodes": existing_validation["episode_reports"],
                "validation": existing_validation,
            }
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("Existing dataset already satisfies the requested pilot contract.")
            print(json.dumps(report, indent=2))
            return 0
        raise DatasetError(
            "dataset root contains existing episodes but does not satisfy the "
            "requested cardinality; use a different dataset root"
        )

    if available < minimum_free:
        raise DatasetError(
            f"collection requires {minimum_free:.1f} GiB free; "
            f"only {available:.1f} GiB is available"
        )

    client = carla.Client(args.host, args.port)
    # A cold UE4 server can expose its TCP port before map loading is ready.
    # Town loading on laptop storage may legitimately exceed 30 seconds.
    client.set_timeout(120.0)
    world = client.get_world()
    current_map = world.get_map().name.rsplit("/", 1)[-1]
    if current_map != args.map:
        world = client.load_world(args.map)
    else:
        print(f"Reusing already-loaded map {current_map}", flush=True)
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    start = time.perf_counter()
    episode_reports: list[dict[str, Any]] = []
    failure: str | None = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = config["simulator"]["fixed_delta_seconds"]
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)

        for episode_number in range(args.episodes):
            print(f"Collecting episode {episode_number + 1}/{args.episodes}", flush=True)
            report = collect_episode(
                args=args,
                config=config,
                world=world,
                traffic_manager=traffic_manager,
                episode_number=episode_number,
                stop_free_gib=stop_free,
            )
            episode_reports.append(report)
            print(
                f"Completed {report['episode_id']}: {report['samples']} samples, "
                f"{report['image_bytes'] / (1024**2):.1f} MiB images",
                flush=True,
            )
    except Exception as error:  # noqa: BLE001 - convert failure into the run report
        failure = f"{type(error).__name__}: {error}"
    finally:
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        world.apply_settings(original_settings)

    validation = validate_dataset(args.dataset_root)
    if validation["status"] != "passed" and failure is None:
        failure = "dataset validation failed"
    if failure is None and (
        validation["episodes"] != args.episodes or validation["samples"] != expected_samples
    ):
        failure = (
            "pilot cardinality mismatch: "
            f"expected {args.episodes} episodes/{expected_samples} samples, "
            f"found {validation['episodes']} episodes/{validation['samples']} samples"
        )
    report = {
        "status": "failed" if failure else "passed",
        "failure": failure,
        "requested_episodes": args.episodes,
        "completed_episodes": len(episode_reports),
        "ticks_per_episode": args.ticks_per_episode,
        "elapsed_wall_seconds": round(time.perf_counter() - start, 3),
        "free_disk_gib_after": round(free_gib(args.dataset_root), 3),
        "episodes": episode_reports,
        "validation": validation,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.dataset_root / "manifest.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
