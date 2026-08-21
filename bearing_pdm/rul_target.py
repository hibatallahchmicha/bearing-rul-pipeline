"""RUL target construction: hours-to-failure, piecewise-capped.

Two decisions are encoded here, both of which change what the model is
being asked to learn.

**Absolute time, not fraction of life.** `rul_model.fraction_of_life_
remaining` is a fine label for *stage* boundaries, but it cannot be a
regression target: computing it requires knowing the bearing's total
lifespan, which is the very quantity being predicted. RUL must be in
absolute units the sensors could in principle support. Snapshot filenames
carry timestamps, so hours-to-failure is directly available.

**Capped, not raw linear.** Early in a bearing's life the vibration
signature does not distinguish "8 days remaining" from "30 days
remaining" -- the bearing just looks healthy, and no feature separates
those cases. Regressing an uncapped linear countdown forces the model to
invent a distinction it cannot observe, so it fits noise in the healthy
regime. Capping makes the target flat while the bearing is healthy and
linear once degradation becomes measurable. This is the piecewise-linear
RUL convention standard in the prognostics literature.

The 168-hour (7-day) default is an operational choice, not a tuned one:
maintenance scheduling runs on a weekly horizon, so resolution beyond one
week has no decision value even where it is theoretically observable.

Known limitation, visible in any leave-one-bearing-out result built on
this target: the four run-to-failure units have lifespans of roughly
360 h, 360 h, 164 h and 1054 h. A model trained on short-lived units and
tested on a long-lived one will systematically under-predict, and vice
versa. This is the same cross-unit transfer problem that limits the alarm
threshold, and it is reported rather than tuned away.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

# Operational planning horizon. Beyond this, remaining life is neither
# reliably observable in the vibration signal nor actionable in scheduling.
DEFAULT_RUL_CAP_HOURS = 168.0  # 7 days


def rul_hours_from_timestamps(timestamps: Sequence[datetime]) -> np.ndarray:
    """Hours remaining until the final observed timestamp.

    The last snapshot of a run-to-failure unit is treated as the failure
    point, giving it an RUL of 0. Only valid for units observed running
    all the way to failure -- a unit whose recording simply stops early
    would get a target that claims it failed when it did not.
    """
    if len(timestamps) < 2:
        raise ValueError(f"need at least 2 timestamps to build an RUL target, got {len(timestamps)}")

    failure_time = max(timestamps)
    hours = np.array([(failure_time - t).total_seconds() / 3600.0 for t in timestamps], dtype=float)

    # Snapshots arrive pre-sorted from ingestion.list_snapshot_files, so
    # `hours` must be strictly decreasing. A repeated or out-of-order
    # timestamp means that ordering broke upstream, which would silently
    # corrupt every RUL label built here. Checking non-negativity instead
    # would be dead code: `failure_time` is the maximum by construction.
    if np.any(np.diff(hours) >= 0):
        raise ValueError("timestamps are not strictly increasing in time")

    return hours


def cap_rul(rul_hours: np.ndarray, cap_hours: float = DEFAULT_RUL_CAP_HOURS) -> np.ndarray:
    """Apply the piecewise-linear cap: flat while healthy, linear near failure."""
    if cap_hours <= 0:
        raise ValueError(f"cap_hours must be positive, got {cap_hours}")
    return np.minimum(rul_hours, cap_hours)


def build_rul_target(
    timestamps: Sequence[datetime], cap_hours: float = DEFAULT_RUL_CAP_HOURS
) -> np.ndarray:
    """Full target pipeline: timestamps -> capped hours-to-failure."""
    return cap_rul(rul_hours_from_timestamps(timestamps), cap_hours)


def aggregate_windows_to_snapshot(
    window_features: dict[str, np.ndarray], windows_per_snapshot: int
) -> dict[str, np.ndarray]:
    """Collapse each snapshot's windows into one row, by median.

    Every window inside a single snapshot spans about one second of real
    time, so all of them share the same remaining life. Left unaggregated,
    one snapshot contributes ~19 near-duplicate rows carrying one
    observation's worth of information -- which inflates the apparent
    sample size by ~19x and lets duplicate rows dominate any variance
    estimate.

    Median rather than mean: a single window containing a transient
    (an impact, an electrical spike) should not move the snapshot's
    summary, and the median is the standard robust choice.
    """
    if windows_per_snapshot < 1:
        raise ValueError(f"windows_per_snapshot must be >= 1, got {windows_per_snapshot}")

    out: dict[str, np.ndarray] = {}
    for name, values in window_features.items():
        n_snapshots = len(values) // windows_per_snapshot
        if n_snapshots == 0:
            raise ValueError(
                f"feature {name!r} has {len(values)} windows, fewer than one snapshot "
                f"of {windows_per_snapshot}"
            )
        trimmed = values[: n_snapshots * windows_per_snapshot]
        out[name] = np.median(trimmed.reshape(n_snapshots, windows_per_snapshot), axis=1)
    return out