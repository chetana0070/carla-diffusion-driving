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

    from carla_diffusion.temporal_bc_dataset import TemporalWindowDataset
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch training dependencies are not installed")
class TemporalBCDatasetTests(unittest.TestCase):
    def test_temporal_shapes_and_clipping(self) -> None:
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
                Image.new("RGB", (640, 360), (index * 20, 64, 32)).save(path)
                paths.append(f"episode/images/{index:08d}.jpg")
            (processed / "report.json").write_text(
                json.dumps({"source_dataset": str(raw)}), encoding="utf-8"
            )
            (processed / "normalization.json").write_text(
                json.dumps(
                    {
                        "state_mean": [1.0] * 10,
                        "state_std": [1e-6] + [2.0] * 9,
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "split": "train",
                "image_paths": paths,
                "state_history": [[3.0] * 10] * 4,
                "condition": [1.0] + [0.0] * 8,
                "action_target": [[0.2, -0.3]] * 16,
                "sample_weight": 1.5,
            }
            (processed / "windows.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            dataset = TemporalWindowDataset(processed, "train", augment=False)
            item = dataset[0]
            self.assertEqual(tuple(item["images"].shape), (4, 3, 224, 224))
            self.assertEqual(tuple(item["state_history"].shape), (4, 10))
            self.assertTrue(bool(torch.all(item["state_history"][:, 0] == 10.0)))
            self.assertTrue(bool(torch.all(item["state_history"][:, 1:] == 1.0)))
            self.assertEqual(tuple(item["condition"].shape), (9,))
            self.assertEqual(tuple(item["target"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
