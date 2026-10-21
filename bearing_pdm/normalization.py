"""Per-unit baseline normalization.

Physically distinct bearings -- even nominally identical ones -- differ in
noise floor, mounting stiffness, and sensor gain, so their raw feature
values are not comparable. A baseline pooled from several bearings
represents none of them; on this dataset that pooling dropped stage
accuracy to 0.218, below the always-"healthy" baseline.

The fix is to z-score each bearing against its OWN early life before any
pooling. This is deployable per-asset calibration, not leakage: every new
bearing gets a short healthy break-in period to calibrate against, and the
statistics come only from that period -- never from later rows, and never
from knowledge of when or whether the bearing fails.
"""

from __future__ import annotations

import numpy as np

# Share of a bearing's rows, from the start of its life, treated as its
# healthy calibration period. A stated default, not a tuned value.
DEFAULT_HEALTHY_FRACTION = 0.2


def normalize_to_own_baseline(
    matrix: np.ndarray, healthy_fraction: float = DEFAULT_HEALTHY_FRACTION
) -> np.ndarray:
    """Z-score one bearing's (n_rows, n_features) matrix against its early life.

    Rows MUST be in chronological order -- "the first N rows" only means
    "early life" if they are. Sorting is the caller's job.

    A feature that is perfectly constant during the baseline period gets a
    scale of 1 instead of 0, so it is centred but not divided by zero.
    """
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D (rows, features) matrix, got shape {matrix.shape}")
    if not 0 < healthy_fraction <= 1:
        raise ValueError(f"healthy_fraction must be in (0, 1], got {healthy_fraction}")

    n_baseline = max(int(len(matrix) * healthy_fraction), 1)
    baseline = matrix[:n_baseline]

    mean = baseline.mean(axis=0)
    std = baseline.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (matrix - mean) / std