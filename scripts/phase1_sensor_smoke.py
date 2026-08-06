#!/usr/bin/env python3
"""Deterministic CARLA camera synchronization smoke test.

The test intentionally stores no images. It verifies server connectivity,
synchronous ticking, sensor dimensions, exact frame alignment, monotonic frame
IDs, actor cleanup, and restoration of asynchronous world settings.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import queue
import statistics
import time
from typing import Any

import carla


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--map", default="Town01")
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fixed-delta", type=float, default=0.1)
    parser.add_argument("--sensor-timeout", type=float, default=30.0)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/evaluations/phase1_smoke_report.json"),
    )
    return parser.parse_args()


def select_vehicle_blueprint(world: carla.World) -> carla.ActorBlueprint:
    library = world.get_blueprint_library()
    preferred = library.filter("vehicle.tesla.model3")
    candidates = preferred or library.filter("vehicle.*")
    if not candidates:
        raise RuntimeError("no vehicle blueprint is available")
    blueprint = sorted(candidates, key=lambda item: item.id)[0]
    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", "hero")
    return blueprint


def spawn_vehicle(world: carla.World, blueprint: carla.ActorBlueprint) -> carla.Vehicle:
    points = sorted(
        world.get_map().get_spawn_points(),
        key=lambda transform: (
            round(transform.location.x, 3),
            round(transform.location.y, 3),
            round(transform.location.z, 3),
        ),
    )
    for transform in points:
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            return actor
    raise RuntimeError("unable to spawn the ego vehicle")


def main() -> int:
    args = parse_args()
    if args.ticks < 1:
        raise ValueError("--ticks must be positive")

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    # Keep the infrastructure smoke test lightweight and deterministic. Larger
    # maps such as Town10HD belong in the frozen evaluation suite.
    world = client.load_world(args.map)
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    actors: list[carla.Actor] = []
    images: queue.Queue[carla.Image] = queue.Queue(maxsize=16)

    start_wall_time = time.perf_counter()
    frame_ids: list[int] = []
    sensor_wait_ms: list[float] = []
    dimensions: Counter[tuple[int, int]] = Counter()
    failure: str | None = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)

        vehicle = spawn_vehicle(world, select_vehicle_blueprint(world))
        actors.append(vehicle)

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))
        camera_bp.set_attribute("fov", "90")
        # Zero means capture on every simulator tick. Matching sensor_tick to
        # fixed_delta can miss frames because the two clocks accumulate
        # floating-point time independently inside Unreal Engine.
        camera_bp.set_attribute("sensor_tick", "0.0")
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=2.2)),
            attach_to=vehicle,
        )
        actors.append(camera)
        camera.listen(images.put)
        vehicle.set_autopilot(True, args.traffic_manager_port)

        for _ in range(args.ticks):
            expected_frame = world.tick()
            wait_start = time.perf_counter()
            try:
                image = images.get(timeout=args.sensor_timeout)
            except queue.Empty as error:
                raise RuntimeError(
                    "camera produced no image for "
                    f"world frame {expected_frame} within "
                    f"{args.sensor_timeout:.1f} seconds"
                ) from error
            sensor_wait_ms.append((time.perf_counter() - wait_start) * 1000)
            frame_ids.append(image.frame)
            dimensions[(image.width, image.height)] += 1
            if image.frame != expected_frame:
                raise RuntimeError(
                    f"camera/world frame mismatch: camera={image.frame}, world={expected_frame}"
                )

        if len(set(frame_ids)) != args.ticks:
            raise RuntimeError("duplicate camera frame IDs detected")
        if any(b - a != 1 for a, b in zip(frame_ids, frame_ids[1:])):
            raise RuntimeError("non-consecutive camera frame IDs detected")
        if set(dimensions) != {(args.width, args.height)}:
            raise RuntimeError(f"unexpected camera dimensions: {dict(dimensions)}")
    except Exception as error:  # report failure after cleanup
        failure = f"{type(error).__name__}: {error}"
    finally:
        for actor in reversed(actors):
            try:
                if "sensor" in actor.type_id:
                    actor.stop()
                actor.destroy()
            except RuntimeError:
                pass
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        world.apply_settings(original_settings)

    elapsed = time.perf_counter() - start_wall_time
    report: dict[str, Any] = {
        "status": "failed" if failure else "passed",
        "failure": failure,
        "carla_server_version": client.get_server_version(),
        "carla_client_version": client.get_client_version(),
        "map": world.get_map().name,
        "sensor_tick_seconds": 0.0,
        "sensor_timeout_seconds": args.sensor_timeout,
        "seed": args.seed,
        "requested_ticks": args.ticks,
        "received_frames": len(frame_ids),
        "unique_frames": len(set(frame_ids)),
        "first_frame": frame_ids[0] if frame_ids else None,
        "last_frame": frame_ids[-1] if frame_ids else None,
        "dimensions": {f"{width}x{height}": count for (width, height), count in dimensions.items()},
        "fixed_delta_seconds": args.fixed_delta,
        "elapsed_wall_seconds": round(elapsed, 3),
        "effective_ticks_per_second": round(len(frame_ids) / elapsed, 3) if elapsed else None,
        "sensor_wait_ms_mean": round(statistics.fmean(sensor_wait_ms), 3) if sensor_wait_ms else None,
        "sensor_wait_ms_max": round(max(sensor_wait_ms), 3) if sensor_wait_ms else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
