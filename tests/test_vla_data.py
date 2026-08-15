import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.vla_data import (
    build_vla_record,
    build_vocabulary,
    decode_condition,
    encode_instruction,
    instruction_for,
)


class VLADataTests(unittest.TestCase):
    def test_condition_decoding_and_instruction(self) -> None:
        route, light = decode_condition((0, 1, 0, 0, 1, 0, 0, 0, 0))
        self.assertEqual((route, light), ("left", "none"))
        self.assertIn("turn left", instruction_for(route, light))

    def test_instruction_encoding_is_fixed_width(self) -> None:
        vocabulary = build_vocabulary()
        token_ids, mask = encode_instruction(
            "Drive safely: follow the current lane.", vocabulary, 20
        )
        self.assertEqual(len(token_ids), 20)
        self.assertEqual(len(mask), 20)
        self.assertEqual(mask[0], 1)
        self.assertEqual(mask[-1], 0)

    def test_vla_record_preserves_split_and_horizon(self) -> None:
        window = {
            "split": "train",
            "episode_id": "episode",
            "route_id": "route",
            "anchor_frame_id": 7,
            "image_paths": ["a.jpg"] * 4,
            "state_history": [[0.0] * 10] * 4,
            "condition": [1, 0, 0, 0, 0, 0, 0, 1, 0],
            "action_target": [[0.1, 0.2]] * 16,
            "sample_weight": 1.0,
            "categories": ["routine"],
        }
        record = build_vla_record(
            window,
            vocabulary=build_vocabulary(),
            planner_horizon=16,
            max_instruction_tokens=20,
        )
        self.assertEqual(record.split, "train")
        self.assertEqual(record.traffic_light_state, "green")
        self.assertEqual(len(record.action_chunk_target), 16)

    def test_non_one_hot_condition_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decode_condition((1, 1, 0, 0, 1, 0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
