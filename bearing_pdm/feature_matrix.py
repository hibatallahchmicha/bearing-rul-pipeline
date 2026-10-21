"""Model inputs from a Gold feature frame -- one implementation, shared.

Evaluation, registration and batch scoring all need identical inputs: the
same row order, the same per-bearing normalization, the same feature
order. Training-serving skew -- a model receiving differently prepared
data in production than it was evaluated on -- most often comes from that
preparation being written more than once and the copies drifting apart.
It is written once here, and every notebook calls it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from bearing_pdm.normalization import DEFAULT_HEALTHY_FRACTION, normalize_to_own_baseline


def build_sensor_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    unit_column: str = "bearing_id",
    time_column: str = "snapshot_ts",
    healthy_fraction: float = DEFAULT_HEALTHY_FRACTION,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sort by unit then time, and normalize each unit against its own early life.

    Returns `(ordered_frame, matrix)`, where row i of `matrix` belongs to
    row i of `ordered_frame`. Callers must use the returned frame -- not the
    one they passed in -- for anything joined back to predictions, because
    the input order is not preserved.
    """
    required = [unit_column, time_column, *feature_columns]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(f"frame is missing columns: {missing}")

    ordered = frame.sort_values([unit_column, time_column], kind="stable").reset_index(drop=True)

    blocks = [
        normalize_to_own_baseline(group[list(feature_columns)].to_numpy(dtype=float), healthy_fraction)
        for _, group in ordered.groupby(unit_column, sort=True)
    ]
    return ordered, np.vstack(blocks)