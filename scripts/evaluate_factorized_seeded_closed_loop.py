#!/usr/bin/env python3
"""Forward an explicit Phase 7 seed to the factorized evaluator."""

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Mapping
from pathlib import Path


def seed_arguments(environment: Mapping[str, str]) -> list[str]:
    raw_seed = environment.get("PHASE7_SEED")
    if raw_seed is None or not raw_seed.strip():
        raise ValueError("PHASE7_SEED must be set for seeded evaluation")
    try:
        seed = int(raw_seed)
    except ValueError as error:
        raise ValueError("PHASE7_SEED must be an integer") from error
    if seed < 0:
        raise ValueError("PHASE7_SEED cannot be negative")
    return ["--seed", str(seed)]


def main() -> int:
    try:
        arguments = seed_arguments(os.environ)
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    if "--seed" in sys.argv:
        print("FAIL: seed was supplied by both the runner and command line", file=sys.stderr)
        return 2
    sys.argv.extend(arguments)
    evaluator = Path(__file__).with_name("evaluate_factorized_closed_loop.py")
    runpy.run_path(str(evaluator), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
