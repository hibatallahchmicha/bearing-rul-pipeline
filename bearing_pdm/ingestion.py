"""Read raw IMS bearing snapshot files and replay them as a simulated live stream.

Each snapshot file is a fixed-length window of raw accelerometer samples
captured at one point in a bearing's run-to-failure life. Downstream stages
(segmentation, features, health index) consume signal arrays; this module
is the only place that touches the filesystem, so the rest of the pipeline
never needs to know files are involved at all.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLES_PER_FILE = 20480
SAMPLE_RATE_HZ = 20_000
TIMESTAMP_FORMAT = "%Y.%m.%d.%H.%M.%S"


def list_snapshot_files(test_dir: Path) -> list[Path]:
    """Return every snapshot file in a test-set directory, oldest first.

    Filenames use NASA's fixed 'YYYY.MM.DD.HH.MM.SS' format, which sorts
    correctly as plain strings -- no date parsing needed for ordering.
    """
    files = [p for p in Path(test_dir).iterdir() if p.is_file()]
    if not files:
        raise FileNotFoundError(f"no snapshot files found in {test_dir}")
    return sorted(files, key=lambda p: p.name)


def read_snapshot(path: Path) -> np.ndarray:
    """Load one snapshot file into an array of shape (n_samples, n_channels)."""
    return np.loadtxt(path)


def _parse_timestamp(name: str) -> datetime:
    return datetime.strptime(name, TIMESTAMP_FORMAT)


def _infer_gap_seconds(files: list[Path]) -> float:
    """Estimate the real recording interval from the first two filenames."""
    if len(files) < 2:
        return 600.0  # NASA's documented ~10-minute cadence, as a fallback
    gap = _parse_timestamp(files[1].name) - _parse_timestamp(files[0].name)
    return max(gap.total_seconds(), 1.0)


def stream_snapshots(test_dir: Path, speed: float = 1000.0) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (timestamp, signal) pairs in chronological order, paced like a live feed.

    `speed` is a playback multiplier against the real recording cadence.
    speed=1.0 replays at the true pace (~10 minutes between snapshots);
    the default (1000x) compresses that into under a second per snapshot,
    which is what you want for development and for demoing drift detection
    without waiting on a bearing's real multi-week lifespan.
    """
    files = list_snapshot_files(test_dir)
    replay_gap = _infer_gap_seconds(files) / speed

    last_yield = None
    for path in files:
        if last_yield is not None:
            wait = replay_gap - (time.monotonic() - last_yield)
            if wait > 0:
                time.sleep(wait)
        last_yield = time.monotonic()
        yield path.name, read_snapshot(path)