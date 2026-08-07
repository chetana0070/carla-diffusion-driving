import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.config import ConfigError, load_and_validate_config


class ConfigTests(unittest.TestCase):
    def test_project_config_is_valid(self) -> None:
        config = load_and_validate_config(ROOT / "configs" / "project.json")
        self.assertEqual(config["policy"]["action_dimension"], 2)
        self.assertEqual(config["preprocessing"]["model_state_dimension"], 10)
        self.assertEqual(config["behavioral_cloning"]["scalar_input_dimension"], 19)
        self.assertEqual(config["behavioral_cloning"]["output_dimension"], 2)
        self.assertEqual(config["behavioral_cloning"]["normalized_state_clip"], 10.0)
        self.assertEqual(config["closed_loop_evaluation"]["episodes"], 3)
        self.assertEqual(config["closed_loop_evaluation"]["seed"], 20260901)
        self.assertEqual(
            config["closed_loop_evaluation"]["expert_oracle_protocol_version"],
            "2.0.0",
        )
        self.assertEqual(config["temporal_behavioral_cloning"]["history_frames"], 4)
        self.assertEqual(config["temporal_behavioral_cloning"]["batch_size"], 8)
        self.assertTrue(config["simulator"]["synchronous_mode"])

    def test_invalid_delta_is_rejected(self) -> None:
        import json
        import tempfile

        config = load_and_validate_config(ROOT / "configs" / "project.json")
        config["simulator"]["fixed_delta_seconds"] = 0.05
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_and_validate_config(path)


if __name__ == "__main__":
    unittest.main()
