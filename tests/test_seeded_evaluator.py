import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_residual_seeded_closed_loop import seed_arguments


class SeededEvaluatorTests(unittest.TestCase):
    def test_explicit_seed_is_forwarded(self) -> None:
        self.assertEqual(
            seed_arguments({"PHASE6_RESIDUAL_SEED": "20260902"}),
            ["--seed", "20260902"],
        )

    def test_missing_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            seed_arguments({})

    def test_non_integer_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            seed_arguments({"PHASE6_RESIDUAL_SEED": "seed-two"})

    def test_environment_contract_uses_exact_variable(self) -> None:
        with patch.dict(os.environ, {"PHASE6_RESIDUAL_SEED": "9"}, clear=True):
            self.assertEqual(seed_arguments(os.environ), ["--seed", "9"])


if __name__ == "__main__":
    unittest.main()
