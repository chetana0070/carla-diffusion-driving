#!/usr/bin/env python3
"""Read-only system readiness check for the CARLA development machine."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def read_os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip().strip('"')
    except OSError:
        pass
    return values


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def memory_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return None


def gpu_report() -> dict[str, Any]:
    query = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query:
        return {"available": False}
    first_line = query.splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 3:
        return {"available": True, "raw": first_line}
    return {
        "available": True,
        "name": parts[0],
        "vram_mib": int(parts[1]),
        "driver_version": parts[2],
    }


def build_report() -> dict[str, Any]:
    os_release = read_os_release()
    usage = shutil.disk_usage(Path.home())
    gpu = gpu_report()
    python_supported = (3, 10) <= sys.version_info[:2] <= (3, 12)
    ubuntu_version = os_release.get("VERSION_ID")
    official_ubuntu = ubuntu_version in {"20.04", "22.04"}
    free_disk_gib = usage.free / 1024**3
    ram_gib = memory_gib()

    checks = {
        "python_supported": python_supported,
        "official_carla_ubuntu": official_ubuntu,
        "minimum_free_disk": free_disk_gib >= 100,
        "recommended_free_disk": free_disk_gib >= 350,
        "minimum_ram": ram_gib is not None and ram_gib >= 15,
        "recommended_ram": ram_gib is not None and ram_gib >= 31,
        "nvidia_gpu": bool(gpu.get("available")),
        # Consumer 8 GB GPUs commonly report slightly below 8192 MiB to nvidia-smi.
        "minimum_vram": int(gpu.get("vram_mib", 0)) >= 8000,
        "port_2000_available": port_available(2000),
        "port_2001_available": port_available(2001),
    }
    return {
        "platform": {
            "system": platform.system(),
            "kernel": platform.release(),
            "architecture": platform.machine(),
            "ubuntu_version": ubuntu_version,
            "python": platform.python_version(),
        },
        "resources": {
            "ram_gib": round(ram_gib, 1) if ram_gib is not None else None,
            "home_free_disk_gib": round(free_disk_gib, 1),
            "gpu": gpu,
        },
        "checks": checks,
        "blocking_failures": [
            name
            for name in (
                "python_supported",
                "minimum_free_disk",
                "minimum_ram",
                "nvidia_gpu",
                "minimum_vram",
                "port_2000_available",
                "port_2001_available",
            )
            if not checks[name]
        ],
        "warnings": [
            name
            for name in ("official_carla_ubuntu", "recommended_free_disk", "recommended_ram")
            if not checks[name]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + os.linesep, encoding="utf-8")
    return 1 if report["blocking_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
