"""Leakage-resistant deterministic route splitting."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class RouteRecord:
    route_id: str
    town: str
    weather: str
    traffic_seed: int


def split_routes(
    routes: Iterable[RouteRecord],
    train_towns: set[str],
    validation_towns: set[str],
    test_towns: set[str],
) -> dict[str, list[RouteRecord]]:
    """Assign entire routes by town and reject leakage or unknown towns."""

    if train_towns & validation_towns or train_towns & test_towns or validation_towns & test_towns:
        raise ValueError("town split definitions overlap")

    output: dict[str, list[RouteRecord]] = {"train": [], "validation": [], "test": []}
    seen_route_ids: set[str] = set()
    for route in routes:
        if route.route_id in seen_route_ids:
            raise ValueError(f"duplicate route_id: {route.route_id}")
        seen_route_ids.add(route.route_id)

        if route.town in train_towns:
            split = "train"
        elif route.town in validation_towns:
            split = "validation"
        elif route.town in test_towns:
            split = "test"
        else:
            raise ValueError(f"town is not assigned to a split: {route.town}")
        output[split].append(route)

    for values in output.values():
        values.sort()
    return output


def split_pilot_routes(
    route_ids: Iterable[str],
    *,
    seed: int,
    validation_routes: int = 1,
    test_routes: int = 1,
) -> dict[str, list[str]]:
    """Create a deterministic route-level split for a single-town pipeline pilot."""
    route_list = list(route_ids)
    unique = sorted(set(route_list))
    if len(unique) != len(route_list):
        raise ValueError("duplicate route_id")
    if validation_routes < 1 or test_routes < 1:
        raise ValueError("validation_routes and test_routes must be positive")
    if len(unique) <= validation_routes + test_routes:
        raise ValueError("not enough routes to create non-empty train/validation/test splits")

    ordered = sorted(
        unique,
        key=lambda route_id: hashlib.sha256(f"{seed}:{route_id}".encode()).hexdigest(),
    )
    validation_end = validation_routes
    test_end = validation_end + test_routes
    return {
        "train": sorted(ordered[test_end:]),
        "validation": sorted(ordered[:validation_end]),
        "test": sorted(ordered[validation_end:test_end]),
    }
