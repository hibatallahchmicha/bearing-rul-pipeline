"""RUL regression metrics, with an asymmetric score as the headline.

RMSE and MAE treat a prediction that is 20 hours early and one that is 20
hours late as identical errors. In maintenance they are not remotely
identical: an early prediction costs a technician an inspection that
turns out to be unnecessary, while a late prediction means the bearing
failed in service -- unplanned downtime, emergency repair, and whatever
the failure took with it. Any metric that scores those the same is
scoring the wrong thing.

The asymmetric score applies an exponential penalty in both directions,
steeper on the late side, following the C-MAPSS prognostics scoring
convention. The convention's constants (13 early, 10 late) are calibrated
to a 125-cycle target range; they are rescaled here in proportion to the
RUL cap so the penalty keeps the same shape relative to the target's
actual range rather than saturating.

Deliberately NOT included: R-squared. On a monotonically decreasing
target, R-squared is inflated by the trend alone -- a model that has
learned nothing except "remaining life decreases over time" scores well
on it without detecting degradation at all. Comparison against the
baselines in `baselines.py` is what separates a real model from a clock,
and that comparison is the point of reporting these numbers at all.

The score is a SUM, not a mean, so it is only comparable between
evaluations with the same number of predictions -- which is why `n` is
carried alongside it in `RULMetrics`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bearing_pdm.rul_target import DEFAULT_RUL_CAP_HOURS

# C-MAPSS scoring constants, defined against its 125-cycle target range.
CMAPSS_EARLY_SCALE = 13.0
CMAPSS_LATE_SCALE = 10.0
CMAPSS_REFERENCE_RANGE = 125.0
# Operational choice, not a tuned value: 48 hours is enough lead time to
# schedule a technician and order a replacement bearing.
DEFAULT_ALERT_HORIZON_HOURS = 48.0

# Guards the exponential against overflow on a pathological prediction.
MAX_SCORE_EXPONENT = 50.0


@dataclass(frozen=True)
class RULMetrics:
    """All numbers in hours except `asymmetric_score` (unitless) and `n`."""

    asymmetric_score: float
    late_fraction: float
    mean_late_error: float
    mae: float
    rmse: float
    n: int


def asymmetric_score(
    y_true: np.ndarray, y_pred: np.ndarray, cap_hours: float = DEFAULT_RUL_CAP_HOURS
) -> float:
    """Exponential penalty, steeper for late predictions than early ones.

    d = predicted - actual. d > 0 means the model claimed MORE remaining
    life than the bearing had -- the dangerous direction.
    """
    scale = cap_hours / CMAPSS_REFERENCE_RANGE
    early_scale = CMAPSS_EARLY_SCALE * scale
    late_scale = CMAPSS_LATE_SCALE * scale

    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    exponent = np.where(d < 0, -d / early_scale, d / late_scale)
    exponent = np.clip(exponent, 0.0, MAX_SCORE_EXPONENT)
    return float(np.sum(np.exp(exponent) - 1.0))


def rul_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, cap_hours: float = DEFAULT_RUL_CAP_HOURS
) -> RULMetrics:
    """Compute the full RUL metric set for one set of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on an empty prediction set")

    error = y_pred - y_true
    is_late = error > 0

    return RULMetrics(
        asymmetric_score=asymmetric_score(y_true, y_pred, cap_hours),
        late_fraction=float(np.mean(is_late)),
        mean_late_error=float(np.mean(error[is_late])) if np.any(is_late) else 0.0,
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        n=int(y_true.size),
    )
def provable_false_alarm_rate(
    predicted_rul: np.ndarray,
    hours_to_end: np.ndarray,
    alert_horizon_hours: float = DEFAULT_ALERT_HORIZON_HOURS,
) -> float:
    """Share of alerts raised when the bearing provably had more time left.

    `hours_to_end` is the time from each snapshot to the end of that
    bearing's recording. For a bearing that failed, this IS its remaining
    life. For a bearing still running when the test stopped, it is only a
    lower bound -- the label is right-censored. In both cases, a snapshot
    with more than `alert_horizon_hours` to go provably had more than the
    horizon left, so an alert there is provably false.

    This is what lets bearings that never failed -- which have no RUL label
    at all -- still be used to measure false alarms.

    Returns NaN when no snapshot is far enough from the end to judge.
    NaN fails every comparison, so a promotion gate checking
    `rate <= limit` fails safe instead of passing on no evidence.
    """
    predicted_rul = np.asarray(predicted_rul, dtype=float)
    hours_to_end = np.asarray(hours_to_end, dtype=float)

    if predicted_rul.shape != hours_to_end.shape:
        raise ValueError(f"shape mismatch: predicted {predicted_rul.shape}, hours_to_end {hours_to_end.shape}")
    if alert_horizon_hours <= 0:
        raise ValueError(f"alert_horizon_hours must be positive, got {alert_horizon_hours}")

    provably_safe = hours_to_end > alert_horizon_hours
    if not provably_safe.any():
        return float("nan")
    return float(np.mean(predicted_rul[provably_safe] <= alert_horizon_hours))