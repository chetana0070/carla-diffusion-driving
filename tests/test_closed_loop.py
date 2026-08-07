import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.closed_loop import (
    NoProgressMonitor,
    aggregate_episode_reports,
    evaluate_expert_oracle_gate,
    latency_summary,
    longitudinal_to_pedals,
    route_command_from_geometry,
)


class ClosedLoopTests(unittest.TestCase):
    def test_no_progress_monitor_waits_for_full_window(self) -> None:
        monitor = NoProgressMonitor(window_ticks=4, minimum_distance_m=2.0)
        self.assertFalse(monitor.update(0.0))
        self.assertFalse(monitor.update(0.5))
        self.assertFalse(monitor.update(1.0))
        self.assertTrue(monitor.update(1.5))

    def test_no_progress_monitor_accepts_sufficient_motion_and_reset(self) -> None:
        monitor = NoProgressMonitor(window_ticks=3, minimum_distance_m=1.0)
        self.assertFalse(monitor.update(0.0))
        self.assertFalse(monitor.update(0.5))
        self.assertFalse(monitor.update(1.5))
        monitor.reset()
        self.assertFalse(monitor.update(10.0))

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

    def test_expert_oracle_composite_gate(self) -> None:
        aggregate = {
            "mean_route_progress_fraction": 0.23,
            "mean_distance_traveled_m": 208.0,
            "total_collisions": 0,
            "total_lane_invasions": 0,
            "total_red_light_violations": 0,
        }
        contract = {
            "expert_oracle_protocol_version": "2.0.0",
            "expert_oracle_min_route_progress_fraction": 0.2,
            "expert_oracle_min_mean_distance_m": 150.0,
            "expert_oracle_max_collisions": 0,
            "expert_oracle_max_lane_invasions": 0,
            "expert_oracle_max_red_light_violations": 0,
            "expert_oracle_max_no_progress_terminations": 0,
        }
        gate = evaluate_expert_oracle_gate(aggregate, {"tick_limit": 3}, contract)
        self.assertTrue(gate["passed"])
        aggregate["total_collisions"] = 1
        gate = evaluate_expert_oracle_gate(aggregate, {"tick_limit": 3}, contract)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["collision_gate_passed"])


if __name__ == "__main__":
    unittest.main()
