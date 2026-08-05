"""Leakage-resistant deterministic route splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


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

