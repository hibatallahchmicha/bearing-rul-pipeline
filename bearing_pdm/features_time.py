"""Time-domain statistics per window.

Each is a standard vibration-analysis indicator (ISO 13373-1 lists RMS,
crest factor, and kurtosis as baseline condition-monitoring statistics).
Kurtosis is the one worth defending in detail: healthy vibration is close
to Gaussian (kurtosis ~= 3); a localized defect (a pit/spall) produces
sharp impacts that fatten the distribution's tails long before overall
energy (RMS) moves. This is the earliest time-domain warning sign in the
bearing-diagnostics literature (Dyer & Stewart, 1978).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def time_domain_features(windows: np.ndarray) -> dict[str, np.ndarray]:
    """Compute time-domain statistics for each row of `windows`.

    Parameters
    ----------
    windows : array of shape (n_windows, window_size)

    Returns a dict of feature_name -> array of shape (n_windows,).
    """
    rms = np.sqrt(np.mean(windows**2, axis=1))
    std = np.std(windows, axis=1)
    peak = np.max(np.abs(windows), axis=1)
    peak_to_peak = np.ptp(windows, axis=1)
    crest_factor = np.divide(peak, rms, out=np.zeros_like(peak), where=rms > 0)

    return {
        "rms": rms,
        "std": std,
        "kurtosis": kurtosis(windows, axis=1, fisher=False),
        "skewness": skew(windows, axis=1),
        "crest_factor": crest_factor,
        "peak_to_peak": peak_to_peak,
    }