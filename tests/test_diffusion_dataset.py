import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch

    from carla_diffusion.diffusion_dataset import DiffusionWindowDataset
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class DiffusionDatasetTests(unittest.TestCase):
    def test_returns_complete_action_chunk(self) -> None:
        assert torch is not None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            raw = root / "raw"
            processed.mkdir()
            paths = []
            for index in range(4):
                path = raw / "episode" / "images" / f"{index:08d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 36), (index * 20, 64, 32)).save(path)
                paths.append(f"episode/images/{index:08d}.jpg")
            (processed / "report.json").write_text(
                json.dumps({"source_dataset": str(raw)}), encoding="utf-8"
            )
            (processed / "normalization.json").write_text(
                json.dumps({"state_mean": [0.0] * 10, "state_std": [1.0] * 10}),
                encoding="utf-8",
            )
            actions = [[index / 20, -index / 20] for index in range(16)]
            row = {
                "split": "train",
                "image_paths": paths,
                "state_history": [[0.0] * 10] * 4,
                "condition": [1.0] + [0.0] * 8,
                "action_target": actions,
                "sample_weight": 1.5,
            }
            (processed / "windows.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            item = DiffusionWindowDataset(
                processed, "train", action_horizon=16, image_size=32
            )[0]
            self.assertEqual(tuple(item["images"].shape), (4, 3, 32, 32))
            self.assertEqual(tuple(item["target"].shape), (16, 2))
            self.assertTrue(torch.allclose(item["target"], torch.tensor(actions)))

    def test_rejects_wrong_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.json").write_text(
                json.dumps({"source_dataset": directory}), encoding="utf-8"
            )
            (root / "normalization.json").write_text(
                json.dumps({"state_mean": [0.0] * 10, "state_std": [1.0] * 10}),
                encoding="utf-8",
            )
            row = {
                "split": "train",
                "image_paths": ["missing.jpg"] * 4,
                "state_history": [[0.0] * 10] * 4,
                "condition": [0.0] * 9,
                "action_target": [[0.0, 0.0]] * 15,
                "sample_weight": 1.0,
            }
            (root / "windows.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                DiffusionWindowDataset(root, "train", action_horizon=16)


if __name__ == "__main__":
    unittest.main()
