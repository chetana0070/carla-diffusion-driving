import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.audit import (
    evaluate_readiness,
    numeric_summary,
    quantile,
    usable_temporal_windows,
)


class AuditTests(unittest.TestCase):
    @staticmethod
    def readiness_metrics() -> dict:
        return {
            "samples": 6000,
            "episodes": 10,
            "usable_temporal_windows": 5820,
            "image_integrity": {
                "failure_count": 0,
                "dimensions": {"640x360": 6000},
                "formats": {"JPEG": 6000},
            },
            "actions": {
                "absolute_steering": {"p95": 0.1},
                "active_braking_fraction": 0.02,
                "stationary_brake_hold_fraction": 0.1,
            },
            "states": {
                "moving_fraction": 0.8,
                "lead_vehicle_available_fraction": 0.1,
                "lead_vehicle_distance_outlier_count": 0,
                "traffic_light_available_fraction": 0.1,
                "acceleration_outlier_fraction": 0.0,
            },
            "categorical": {
                "route_command": {
                    name: {"count": 20} for name in ("left", "right", "straight")
                }
            },
            "events": {"collision": {"fraction": 0.0}},
        }

    def test_quantile_uses_linear_interpolation(self) -> None:
        self.assertEqual(quantile([0.0, 10.0], 0.25), 2.5)
        self.assertEqual(quantile([3.0], 0.95), 3.0)

    def test_numeric_summary(self) -> None:
        summary = numeric_summary([1.0, 2.0, 3.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["median"], 2.0)
        self.assertEqual(summary["mean"], 2.0)

    def test_temporal_window_count(self) -> None:
        self.assertEqual(usable_temporal_windows([600] * 10, history=4, horizon=16), 5820)
        self.assertEqual(usable_temporal_windows([10], history=4, horizon=16), 0)

    def test_lead_distance_outlier_fails_required_gate(self) -> None:
        metrics = self.readiness_metrics()
        metrics["states"]["lead_vehicle_distance_outlier_count"] = 1
        gates = {gate["name"]: gate for gate in evaluate_readiness(metrics)}
        self.assertEqual(gates["lead_vehicle_distance_integrity"]["status"], "failed")
        self.assertTrue(gates["lead_vehicle_distance_integrity"]["required"])

    def test_stationary_hold_is_not_active_braking(self) -> None:
        metrics = self.readiness_metrics()
        metrics["actions"]["active_braking_fraction"] = 0.0
        metrics["actions"]["stationary_brake_hold_fraction"] = 0.5
        gates = {gate["name"]: gate for gate in evaluate_readiness(metrics)}
        self.assertEqual(gates["active_braking_coverage"]["status"], "failed")
        self.assertEqual(gates["stationary_brake_hold_balance"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
