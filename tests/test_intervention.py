import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.intervention import InterventionMonitor, InterventionThresholds


def thresholds() -> InterventionThresholds:
    return InterventionThresholds(
        warmup_ticks=3,
        lane_offset_m=0.7,
        heading_error_degrees=15.0,
        unsafe_lead_distance_m=6.0,
        unsafe_lead_min_longitudinal=0.05,
        stall_ticks=3,
        stationary_speed_mps=0.5,
    )


class InterventionTests(unittest.TestCase):
    def test_warmup_suppresses_early_trigger(self) -> None:
        monitor = InterventionMonitor(thresholds())
        state = [2.0, 0.0, 8.0, 1.0, 0.0, -1.0, 0.0, -1.0]
        self.assertEqual(monitor.update(state, "none", 0.0), ())
        self.assertEqual(monitor.update(state, "none", 0.0), ())
        self.assertEqual(monitor.update(state, "none", 0.0), ("lane_offset",))

    def test_geometric_and_lead_triggers_are_reported(self) -> None:
        monitor = InterventionMonitor(thresholds())
        state = [4.0, 0.0, 8.0, 0.8, math.radians(20), 5.0, -2.0, -1.0]
        monitor.update(state, "none", 0.2)
        monitor.update(state, "none", 0.2)
        reasons = monitor.update(state, "none", 0.2)
        self.assertEqual(
            reasons,
            ("lane_offset", "heading_error", "unsafe_lead_approach"),
        )

    def test_red_light_resets_stall_counter(self) -> None:
        monitor = InterventionMonitor(thresholds())
        state = [0.0, 0.0, 8.0, 0.0, 0.0, -1.0, 0.0, 5.0]
        for _ in range(4):
            self.assertNotIn("stalled", monitor.update(state, "red", -0.5))
        self.assertNotIn("stalled", monitor.update(state, "none", 0.0))
        self.assertNotIn("stalled", monitor.update(state, "none", 0.0))
        self.assertIn("stalled", monitor.update(state, "none", 0.0))


if __name__ == "__main__":
    unittest.main()
