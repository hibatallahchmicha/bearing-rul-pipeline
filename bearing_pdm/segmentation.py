"""Cut a raw vibration signal into fixed-length, overlapping windows.

Downstream feature extraction (time- and frequency-domain) operates per
window, not per snapshot -- this is what turns one 20480-sample snapshot
into ~18 independent samples of the bearing's health state, and what gives
the frequency-domain features (features_freq.py) an FFT resolution fine
enough to separate the bearing's characteristic fault frequencies
(15-300 Hz apart) instead of blurring them together.
"""

from __future__ import annotations

import numpy as np

WINDOW_SIZE = 2048   # samples/window -> ~9.8 Hz FFT bin width at 20 kHz
OVERLAP = 0.5         # fraction of each window shared with the next


def window_signal(signal: np.ndarray, window_size: int = WINDOW_SIZE, overlap: float = OVERLAP) -> np.ndarray:
    """Split a 1-D signal into overlapping fixed-length windows.

    Returns an array of shape (n_windows, window_size). Any trailing
    samples that don't fill a full final window are dropped.
    """
    if signal.ndim != 1:
        raise ValueError(f"expected a 1-D single-channel signal, got shape {signal.shape}")

    n = len(signal)
    if n < window_size:
        raise ValueError(f"signal length {n} is shorter than window_size {window_size}")

    hop = int(window_size * (1 - overlap))
    n_windows = (n - window_size) // hop + 1
    starts = np.arange(n_windows) * hop
    return np.stack([signal[s : s + window_size] for s in starts])