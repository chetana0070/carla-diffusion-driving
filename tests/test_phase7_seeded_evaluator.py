import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_factorized_seeded_closed_loop import seed_arguments


class Phase7SeededEvaluatorTests(unittest.TestCase):
    def test_seed_is_forwarded(self) -> None:
        self.assertEqual(seed_arguments({"PHASE7_SEED": "20260902"}), ["--seed", "20260902"])

    def test_missing_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            seed_arguments({})

    def test_negative_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            seed_arguments({"PHASE7_SEED": "-1"})


if __name__ == "__main__":
    unittest.main()
