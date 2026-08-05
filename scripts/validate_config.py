#!/usr/bin/env python3
"""Validate the frozen Phase 0 configuration."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carla_diffusion.config import ConfigError, load_and_validate_config  # noqa: E402


def main() -> int:
    try:
        config = load_and_validate_config(ROOT / "configs" / "project.json")
    except (ConfigError, OSError, ValueError) as error:
        print(f"FAIL: {error}")
        return 1
    print(
        "PASS: configuration is internally consistent "
        f"(CARLA {config['simulator']['carla_version']}, "
        f"{config['simulator']['control_hz']} Hz, "
        f"horizon={config['policy']['action_horizon']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

