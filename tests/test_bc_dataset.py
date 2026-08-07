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

    from carla_diffusion.bc_dataset import SingleFrameWindowDataset
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class BCDatasetTests(unittest.TestCase):
    def test_single_frame_shapes_and_normalization(self) -> None:
        assert torch is not None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            raw = root / "raw"
            image_path = raw / "episode" / "images" / "00000001.jpg"
            processed.mkdir()
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (640, 360), (128, 64, 32)).save(image_path)
            (processed / "report.json").write_text(
                json.dumps({"source_dataset": str(raw)}), encoding="utf-8"
            )
            (processed / "normalization.json").write_text(
                json.dumps({"state_mean": [1.0] * 10, "state_std": [2.0] * 10}),
                encoding="utf-8",
            )
            row = {
                "split": "train",
                "image_paths": ["episode/images/00000001.jpg"] * 4,
                "state_history": [[3.0] * 10] * 4,
                "condition": [1.0] + [0.0] * 8,
                "action_target": [[0.2, -0.3]] * 16,
                "sample_weight": 1.5,
            }
            (processed / "windows.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            dataset = SingleFrameWindowDataset(processed, "train", augment=False)
            item = dataset[0]
            self.assertEqual(tuple(item["image"].shape), (3, 224, 224))
            self.assertEqual(tuple(item["scalar_context"].shape), (19,))
            self.assertTrue(bool(torch.allclose(item["scalar_context"][:10], torch.ones(10))))
            self.assertEqual(tuple(item["target"].shape), (2,))
            self.assertAlmostEqual(float(item["weight"]), 1.5)


if __name__ == "__main__":
    unittest.main()
