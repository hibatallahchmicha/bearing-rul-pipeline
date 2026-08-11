"""Drift detection: is the incoming feature distribution still consistent
with what the health index / stage thresholds were fit on?

Every module built so far -- the health-index baseline, the stage
thresholds, the cost threshold -- is fit ONCE on training data and then
trusted for every window that arrives afterward. In a real deployment
that trust decays: a sensor gets recalibrated, a bearing batch changes,
ambient conditions shift. None of that is bearing degradation, but all of
it can silently invalidate the health-index baseline -- the system would
start raising false alarms (or missing real ones) for reasons unrelated
to actual wear, and nothing built so far would notice. This module is
that notice.

Per-feature two-sample Kolmogorov-Smirnov tests compare a reference batch
against a current batch. KS is nonparametric -- it doesn't assume the
features are Gaussian, which several (spectral energies especially) are
not -- and it naturally reports one result per feature, so drift is
"which features moved," not one opaque flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp

# Per-feature significance level. Testing ~15 features at once means some
# will cross any single p-value threshold by chance alone -- rather than
# correcting for multiple testing (e.g. Bonferroni), DRIFTED_FEATURE_
# FRACTION_ALARM requires a substantial fraction of features to drift
# together before raising an alarm -- cruder, but a sturdier guard
# against one noisy feature triggering a false alert.
P_VALUE_THRESHOLD = 0.01
DRIFTED_FEATURE_FRACTION_ALARM = 0.3


@dataclass(frozen=True)
class DriftReport:
    drifted_features: list[str]
    n_features: int
    drift_detected: bool


def detect_drift(reference: dict[str, np.ndarray], current: dict[str, np.ndarray]) -> DriftReport:
    """Compare each feature's current-batch distribution to its reference.

    `reference` and `current` are feature-name -> array dicts (matching
    features_time.py / features_freq.py output, merged) and must share
    the same feature names.
    """
    drifted = []
    for name, ref_values in reference.items():
        _, p_value = ks_2samp(ref_values, current[name])
        if p_value < P_VALUE_THRESHOLD:
            drifted.append(name)

    fraction_drifted = len(drifted) / len(reference)
    return DriftReport(
        drifted_features=drifted,
        n_features=len(reference),
        drift_detected=fraction_drifted >= DRIFTED_FEATURE_FRACTION_ALARM,
    )


def should_trigger_retrain(recent_reports: list[DriftReport], min_consecutive: int = 3) -> bool:
    """Retrain trigger stub: fires only after `min_consecutive` consecutive
    drift-positive batches, not a single one.

    Deliberately does NOT call a training function. Retraining redefines
    what "healthy" means, which should never happen unattended on the
    first drift flag -- a single degrading bearing in the retrain window
    could poison the new baseline. In a real deployment, this function's
    return value is the trigger for a human-reviewed retrain job.
    """
    if len(recent_reports) < min_consecutive:
        return False
    return all(r.drift_detected for r in recent_reports[-min_consecutive:])