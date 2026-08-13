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
class CorrectiveTrainingTests(unittest.TestCase):
    def test_window_can_override_default_source_dataset(self) -> None:
        assert torch is not None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            base = root / "base"
            correction = root / "correction"
            processed.mkdir()
            paths = []
            for index in range(4):
                relative = f"episode/images/{index:08d}.jpg"
                path = correction / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (index, 2, 3)).save(path)
                paths.append(relative)
            (processed / "report.json").write_text(
                json.dumps({"source_dataset": str(base)}), encoding="utf-8"
            )
            (processed / "normalization.json").write_text(
                json.dumps({"state_mean": [0.0] * 10, "state_std": [1.0] * 10}),
                encoding="utf-8",
            )
            row = {
                "split": "correction_validation",
                "source_dataset": str(correction),
                "image_paths": paths,
                "state_history": [[0.0] * 10] * 4,
                "condition": [1.0] + [0.0] * 8,
                "action_target": [[0.0, 0.0]] * 16,
                "sample_weight": 3.0,
            }
            (processed / "windows.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            item = TemporalWindowDataset(processed, "correction_validation")[0]
            self.assertEqual(tuple(item["images"].shape), (4, 3, 224, 224))
            self.assertEqual(float(item["weight"]), 3.0)


if __name__ == "__main__":
    unittest.main()
