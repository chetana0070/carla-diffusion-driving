import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.splits import RouteRecord, split_routes


class SplitTests(unittest.TestCase):
    def test_routes_are_assigned_without_frame_leakage(self) -> None:
        routes = [
            RouteRecord("r-train", "Town01", "ClearNoon", 1),
            RouteRecord("r-validation", "Town02", "WetNoon", 2),
            RouteRecord("r-test", "Town05", "HardRainNoon", 3),
        ]
        splits = split_routes(routes, {"Town01"}, {"Town02"}, {"Town05"})
        self.assertEqual([route.route_id for route in splits["test"]], ["r-test"])

    def test_duplicate_route_id_is_rejected(self) -> None:
        routes = [
            RouteRecord("duplicate", "Town01", "ClearNoon", 1),
            RouteRecord("duplicate", "Town02", "WetNoon", 2),
        ]
        with self.assertRaises(ValueError):
            split_routes(routes, {"Town01"}, {"Town02"}, {"Town05"})


if __name__ == "__main__":
    unittest.main()
