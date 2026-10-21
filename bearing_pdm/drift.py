"""Drift detection: is incoming data still consistent with what the model
was trained on?

Every fitted component -- health baseline, thresholds, the registered
model -- is trusted for every window that arrives afterward. A recalibrated
sensor, a different bearing type, or a pipeline change can invalidate that
silently. This module is that check.

**Effect size, not significance.** An earlier version flagged features
with a two-sample KS test at p < 0.01. On real data it flagged 16 of 17
features with no drift present: with thousands of snapshots per side, a KS
test detects differences far too small to matter, because two physical
bearings never share an exactly identical distribution. The question a
monitor must answer is "did it move enough to matter?" -- an effect size.

The Population Stability Index measures that: the reference is split into
quantile bins, and PSI sums how far the current data's share of each bin
has moved. Its conventional reading (from credit-risk model monitoring)
is < 0.1 stable, 0.1-0.25 moderate, > 0.25 a significant shift. The 0.25
threshold here is that published convention, not a value tuned on this
data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PSI_BINS = 10
PSI_DRIFT_THRESHOLD = 0.25

# Floor on bin shares so an empty bin cannot produce log(0). Small enough to
# leave occupied bins untouched.
PSI_EPSILON = 1e-4

# Share of features that must drift together before raising an alarm, so one
# noisy feature cannot trigger an alert on its own.
DRIFTED_FEATURE_FRACTION_ALARM = 0.3


@dataclass(frozen=True)
class DriftReport:
    drifted_features: list[str]
    n_features: int
    drift_detected: bool
    psi: dict[str, float] = field(default_factory=dict)


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, n_bins: int = PSI_BINS
) -> float:
    """PSI of `current` against `reference`, using reference quantile bins.

    The outer bins are open-ended, so values beyond the reference's range
    land in the first or last bin -- a shift outside the reference range
    still registers instead of being dropped.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        raise ValueError("PSI needs non-empty reference and current samples")

    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, n_bins + 1)))
    inner_edges = edges[1:-1]

    n_slots = len(inner_edges) + 1
    ref_counts = np.bincount(np.searchsorted(inner_edges, reference, side="right"), minlength=n_slots)
    cur_counts = np.bincount(np.searchsorted(inner_edges, current, side="right"), minlength=n_slots)

    ref_share = np.clip(ref_counts / ref_counts.sum(), PSI_EPSILON, None)
    cur_share = np.clip(cur_counts / cur_counts.sum(), PSI_EPSILON, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def detect_drift(
    reference: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    psi_threshold: float = PSI_DRIFT_THRESHOLD,
) -> DriftReport:
    """Compare each feature's current distribution to its reference by PSI.

    `reference` and `current` are feature-name -> array dicts sharing the
    same feature names.
    """
    psi = {name: population_stability_index(ref_values, current[name]) for name, ref_values in reference.items()}
    drifted = [name for name, value in psi.items() if value > psi_threshold]

    return DriftReport(
        drifted_features=drifted,
        n_features=len(reference),
        drift_detected=len(drifted) / len(reference) >= DRIFTED_FEATURE_FRACTION_ALARM,
        psi=psi,
    )


def should_trigger_retrain(recent_reports: list[DriftReport], min_consecutive: int = 3) -> bool:
    """Retrain trigger stub: fires only after `min_consecutive` consecutive
    drift-positive reports, not a single one.

    Deliberately does NOT call a training function. Retraining redefines
    what "healthy" means, which should never happen unattended on the first
    drift flag -- a single degrading bearing in the retrain window could
    poison the new baseline. In a real deployment, this function's return
    value is the trigger for a human-reviewed retrain job.
    """
    if len(recent_reports) < min_consecutive:
        return False
    return all(r.drift_detected for r in recent_reports[-min_consecutive:])