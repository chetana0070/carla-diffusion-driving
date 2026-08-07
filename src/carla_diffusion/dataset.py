"""Atomic episode storage and dependency-light dataset validation."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from .schema import SampleValidationError, validate_sample

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class DatasetError(RuntimeError):
    """Raised when an episode cannot be written or validated safely."""


def _safe_id(value: str, name: str) -> str:
    if not _SAFE_ID.fullmatch(value):
        raise DatasetError(f"{name} contains unsafe characters: {value!r}")
    return value


def free_gib(path: Path) -> float:
    """Return usable filesystem capacity in GiB."""
    return shutil.disk_usage(path).free / (1024**3)


@dataclass(frozen=True)
class EpisodeSummary:
    episode_id: str
    route_id: str
    samples: int
    first_frame: int | None
    last_frame: int | None
    image_bytes: int


class AtomicEpisodeWriter:
    """Write an episode to staging and publish it with a single rename."""

    def __init__(
        self,
        dataset_root: Path,
        episode_id: str,
        route_id: str,
        metadata: Mapping[str, Any],
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.episode_id = _safe_id(episode_id, "episode_id")
        self.route_id = _safe_id(route_id, "route_id")
        self.final_dir = self.dataset_root / self.episode_id
        self.stage_dir = self.dataset_root / f".{self.episode_id}.inprogress"
        self.failed_dir = self.dataset_root / f".{self.episode_id}.failed"
        if self.final_dir.exists() or self.stage_dir.exists() or self.failed_dir.exists():
            raise DatasetError(f"episode already exists: {self.episode_id}")

        self.images_dir = self.stage_dir / "images"
        self.images_dir.mkdir(parents=True)
        self._samples_path = self.stage_dir / "samples.jsonl"
        self._samples = self._samples_path.open("x", encoding="utf-8")
        self._metadata = dict(metadata)
        self._count = 0
        self._first_frame: int | None = None
        self._last_frame: int | None = None
        self._image_bytes = 0
        self._closed = False

    def add(self, record: Mapping[str, Any], image_bytes: bytes) -> None:
        if self._closed:
            raise DatasetError("cannot add to a closed episode")
        sample = validate_sample(record)
        if sample.episode_id != self.episode_id or sample.route_id != self.route_id:
            raise DatasetError("sample episode/route identity mismatch")
        if self._last_frame is not None and sample.frame_id <= self._last_frame:
            raise DatasetError("sample frame IDs must be strictly increasing")

        expected_relative = Path(self.episode_id) / "images" / f"{sample.frame_id:08d}.jpg"
        if Path(sample.rgb_front_path) != expected_relative:
            raise DatasetError(
                f"unexpected image path {sample.rgb_front_path!r}; "
                f"expected {expected_relative.as_posix()!r}"
            )
        if not image_bytes:
            raise DatasetError("encoded image is empty")

        image_path = self.images_dir / expected_relative.name
        temporary_path = image_path.with_suffix(".jpg.tmp")
        temporary_path.write_bytes(image_bytes)
        os.replace(temporary_path, image_path)

        self._samples.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self._count += 1
        self._first_frame = sample.frame_id if self._first_frame is None else self._first_frame
        self._last_frame = sample.frame_id
        self._image_bytes += len(image_bytes)

    def finalize(self, extra: Mapping[str, Any] | None = None) -> EpisodeSummary:
        if self._closed:
            raise DatasetError("episode is already closed")
        if self._count == 0:
            raise DatasetError("refusing to publish an empty episode")

        self._samples.flush()
        os.fsync(self._samples.fileno())
        self._samples.close()
        summary = EpisodeSummary(
            episode_id=self.episode_id,
            route_id=self.route_id,
            samples=self._count,
            first_frame=self._first_frame,
            last_frame=self._last_frame,
            image_bytes=self._image_bytes,
        )
        payload = {
            **self._metadata,
            **(dict(extra) if extra else {}),
            "episode_id": self.episode_id,
            "route_id": self.route_id,
            "samples": summary.samples,
            "first_frame": summary.first_frame,
            "last_frame": summary.last_frame,
            "image_bytes": summary.image_bytes,
            "status": "complete",
        }
        metadata_temp = self.stage_dir / "episode.json.tmp"
        metadata_temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(metadata_temp, self.stage_dir / "episode.json")
        os.replace(self.stage_dir, self.final_dir)
        self._closed = True
        return summary

    def abort(self, reason: str) -> None:
        if self._closed:
            return
        self._samples.close()
        failure_path = self.stage_dir / "failure.json"
        failure_path.write_text(json.dumps({"reason": reason}, indent=2) + "\n", encoding="utf-8")
        os.replace(self.stage_dir, self.failed_dir)
        self._closed = True


def validate_dataset(dataset_root: Path) -> dict[str, Any]:
    """Validate all published episodes and return a machine-readable report."""
    root = dataset_root.resolve()
    issues: list[str] = []
    episode_reports: list[dict[str, Any]] = []
    route_ids: set[str] = set()
    total_samples = 0
    total_image_bytes = 0

    if not root.is_dir():
        return {
            "status": "failed",
            "dataset_root": str(root),
            "issues": ["dataset root does not exist"],
        }

    completed_directories = (
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    for episode_dir in sorted(completed_directories):
        samples_path = episode_dir / "samples.jsonl"
        metadata_path = episode_dir / "episode.json"
        local_issues: list[str] = []
        frames: list[int] = []
        image_bytes = 0
        route_id: str | None = None
        metadata: dict[str, Any] | None = None

        if not metadata_path.is_file():
            local_issues.append("missing episode.json")
        else:
            try:
                loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not isinstance(loaded_metadata, dict):
                    raise DatasetError("episode.json must contain an object")
                metadata = loaded_metadata
                if metadata.get("status") != "complete":
                    raise DatasetError("episode status is not complete")
            except (DatasetError, json.JSONDecodeError) as error:
                local_issues.append(f"invalid episode.json: {error}")
        if not samples_path.is_file():
            local_issues.append("missing samples.jsonl")
        else:
            lines = samples_path.read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(lines, 1):
                try:
                    record = json.loads(line)
                    sample = validate_sample(record)
                    if sample.episode_id != episode_dir.name:
                        raise DatasetError("episode_id does not match directory")
                    route_id = sample.route_id if route_id is None else route_id
                    if sample.route_id != route_id:
                        raise DatasetError("multiple route IDs in one episode")
                    image_path = root / sample.rgb_front_path
                    if not image_path.is_file():
                        raise DatasetError(f"missing image: {sample.rgb_front_path}")
                    image_bytes += image_path.stat().st_size
                    frames.append(sample.frame_id)
                except (
                    DatasetError,
                    SampleValidationError,
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    AttributeError,
                ) as error:
                    local_issues.append(f"line {line_number}: {error}")

        if frames and any(current <= previous for previous, current in pairwise(frames)):
            local_issues.append("frame IDs are not strictly increasing")
        if metadata is not None:
            expected_metadata = {
                "episode_id": episode_dir.name,
                "route_id": route_id,
                "samples": len(frames),
                "first_frame": frames[0] if frames else None,
                "last_frame": frames[-1] if frames else None,
                "image_bytes": image_bytes,
            }
            for field, expected in expected_metadata.items():
                if metadata.get(field) != expected:
                    local_issues.append(
                        f"episode.json {field} mismatch: "
                        f"expected {expected!r}, found {metadata.get(field)!r}"
                    )
        if route_id in route_ids:
            local_issues.append(f"duplicate route_id across episodes: {route_id}")
        if route_id is not None:
            route_ids.add(route_id)
        issues.extend(f"{episode_dir.name}: {issue}" for issue in local_issues)
        total_samples += len(frames)
        total_image_bytes += image_bytes
        episode_reports.append(
            {
                "episode_id": episode_dir.name,
                "route_id": route_id,
                "samples": len(frames),
                "first_frame": frames[0] if frames else None,
                "last_frame": frames[-1] if frames else None,
                "image_bytes": image_bytes,
                "status": "failed" if local_issues else "passed",
            }
        )

    if not episode_reports:
        issues.append("no completed episodes found")
    return {
        "status": "failed" if issues else "passed",
        "dataset_root": str(root),
        "episodes": len(episode_reports),
        "samples": total_samples,
        "image_bytes": total_image_bytes,
        "episode_reports": episode_reports,
        "issues": issues,
    }
