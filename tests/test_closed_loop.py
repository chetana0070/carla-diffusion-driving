import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.closed_loop import (
    aggregate_episode_reports,
    latency_summary,
    longitudinal_to_pedals,
    route_command_from_geometry,
)


class ClosedLoopTests(unittest.TestCase):
    def test_route_commands_handle_wraparound(self) -> None:
        self.assertEqual(
            route_command_from_geometry(170.0, -170.0, junction_ahead=True),
            "right",
        )
        self.assertEqual(
            route_command_from_geometry(-170.0, 170.0, junction_ahead=True),
            "left",
        )
        self.assertEqual(
            route_command_from_geometry(0.0, 2.0, junction_ahead=False),
            "follow_lane",
        )
        self.assertEqual(
            route_command_from_geometry(0.0, 90.0, junction_ahead=False),
            "follow_lane",
        )
        self.assertEqual(
            route_command_from_geometry(0.0, 2.0, junction_ahead=True),
            "straight",
        )

    def test_longitudinal_action_maps_to_exclusive_pedals(self) -> None:
        self.assertEqual(longitudinal_to_pedals(0.7), (0.7, 0.0))
        self.assertEqual(longitudinal_to_pedals(-0.4), (0.0, 0.4))
        self.assertEqual(longitudinal_to_pedals(2.0), (1.0, 0.0))

    def test_latency_summary(self) -> None:
        summary = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(summary["p50_ms"], 3.0)
        self.assertEqual(summary["p95_ms"], 5.0)
        self.assertEqual(summary["max_ms"], 5.0)

    def test_episode_aggregation(self) -> None:
        base = {
            "route_progress_fraction": 0.5,
            "distance_traveled_m": 20.0,
            "collision_events": 1,
            "lane_invasion_events": 2,
            "red_light_violations": 0,
            "mean_speed_mps": 4.0,
            "mean_abs_steering_rate_per_second": 0.2,
            "mean_abs_longitudinal_rate_per_second": 0.3,
        }
        aggregate = aggregate_episode_reports([base, base])
        self.assertEqual(aggregate["episodes"], 2)
        self.assertEqual(aggregate["total_collisions"], 2)
        self.assertEqual(aggregate["mean_distance_traveled_m"], 20.0)


if __name__ == "__main__":
    unittest.main()
