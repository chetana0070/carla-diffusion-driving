from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.schema import SampleValidationError, validate_sample


def valid_record() -> dict:
    return {
        "schema_version": "1.0.0",
        "episode_id": "ep-001",
        "route_id": "Town01-route-001",
        "frame_id": 42,
        "timestamp_seconds": 4.2,
        "observation": {
            "rgb_front_path": "episodes/ep-001/rgb/000042.jpg",
            "state": [8.2, 0.1, 13.9, -0.03, 0.02, 18.0, -1.2, 32.0],
            "traffic_light_state": "green",
            "route_command": "follow_lane",
        },
        "expert_action": [0.04, 0.22],
        "events": {
            "collision": False,
            "lane_invasion": False,
            "red_light_violation": False,
            "intervention": False,
        },
    }


class SchemaTests(unittest.TestCase):
    def test_valid_sample(self) -> None:
        sample = validate_sample(valid_record())
        self.assertEqual(sample.frame_id, 42)
        self.assertEqual(len(sample.state), 8)

    def test_absolute_image_path_is_rejected(self) -> None:
        record = valid_record()
        record["observation"]["rgb_front_path"] = "/tmp/frame.jpg"
        with self.assertRaises(SampleValidationError):
            validate_sample(record)

    def test_out_of_range_action_is_rejected(self) -> None:
        record = valid_record()
        record["expert_action"] = [1.4, 0.0]
        with self.assertRaises(SampleValidationError):
            validate_sample(record)


if __name__ == "__main__":
    unittest.main()

