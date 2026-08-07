import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.dataset import AtomicEpisodeWriter, DatasetError, validate_dataset


def sample(frame: int, image_path: str) -> dict:
    return {
        "schema_version": "1.0.0",
        "episode_id": "episode-001",
        "route_id": "route-001",
        "frame_id": frame,
        "timestamp_seconds": frame / 10,
        "observation": {
            "rgb_front_path": image_path,
            "state": [1.0, 0.0, 8.0, 0.1, 0.01, -1.0, 0.0, -1.0],
            "traffic_light_state": "none",
            "route_command": "follow_lane",
        },
        "expert_action": [0.0, 0.2],
        "events": {
            "collision": False,
            "lane_invasion": False,
            "red_light_violation": False,
            "intervention": False,
        },
    }


class DatasetTests(unittest.TestCase):
    def test_episode_is_published_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AtomicEpisodeWriter(root, "episode-001", "route-001", {"seed": 7})
            for frame in (10, 11):
                relative = f"episode-001/images/{frame:08d}.jpg"
                writer.add(sample(frame, relative), b"jpeg-placeholder")
            summary = writer.finalize()

            self.assertEqual(summary.samples, 2)
            self.assertFalse((root / ".episode-001.inprogress").exists())
            self.assertTrue((root / "episode-001" / "episode.json").is_file())
            report = validate_dataset(root)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["samples"], 2)

    def test_non_monotonic_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AtomicEpisodeWriter(root, "episode-001", "route-001", {})
            relative = "episode-001/images/00000010.jpg"
            writer.add(sample(10, relative), b"jpeg-placeholder")
            with self.assertRaises(DatasetError):
                writer.add(sample(10, relative), b"jpeg-placeholder")
            writer.abort("test cleanup")

    def test_missing_image_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode-001"
            episode.mkdir()
            (episode / "episode.json").write_text(json.dumps({"status": "complete"}))
            record = sample(10, "episode-001/images/00000010.jpg")
            (episode / "samples.jsonl").write_text(json.dumps(record) + "\n")
            report = validate_dataset(root)
            self.assertEqual(report["status"], "failed")
            self.assertIn("missing image", report["issues"][0])


if __name__ == "__main__":
    unittest.main()
