"""Frequency-domain features per window -- the core signal-processing content.

Time-domain statistics (features_time.py) tell you *that* something
changed; they can't tell you *where* in the bearing it changed, because
RMS/kurtosis collapse the whole window to one number. A spall on the outer
race, inner race, a roller, or the cage each excites a different,
physically-derived frequency (fault_frequencies.py) -- and that
information only survives in the spectrum. Two windows with identical RMS
can have completely different fault signatures; this module is what tells
them apart.

Two complementary spectral views are computed per window:

1. Direct FFT spectrum. Energy concentrated at BPFO/BPFI/BSF/FTF (or their
   harmonics) is direct evidence of a fault at that specific location.

2. Envelope spectrum (via the Hilbert transform). A bearing fault impact
   is a short, broadband mechanical "knock" that excites a high-frequency
   structural resonance of the machine -- it doesn't show up as a clean
   peak at the fault frequency in the raw spectrum, especially early on.
   Instead, the fault's repetition rate shows up as *amplitude
   modulation* riding on top of that resonance. The envelope spectrum
   demodulates the signal (extracts the amplitude envelope, discards the
   high-frequency carrier) so the repetition rate becomes a clean,
   isolated peak. Standard technique for catching rolling-element faults
   early (Randall & Antoni, 2011, "Rolling element bearing diagnostics --
   a tutorial", Mechanical Systems and Signal Processing) -- raw-spectrum
   band energy alone is often too weak to see until a fault is already
   well developed.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from bearing_pdm.fault_frequencies import FaultFrequencies

BAND_HALF_WIDTH_HZ = 5.0   # tolerance around each target frequency
N_HARMONICS = 3             # fault frequency + its 2nd/3rd harmonics


def _band_energy(freqs: np.ndarray, power: np.ndarray, center_hz: float, half_width_hz: float) -> float:
    """Sum spectral power in a small band around one target frequency."""
    mask = (freqs >= center_hz - half_width_hz) & (freqs <= center_hz + half_width_hz)
    return float(power[mask].sum())


def _fault_band_energies(
    freqs: np.ndarray, power: np.ndarray, faults: FaultFrequencies, prefix: str
) -> dict[str, float]:
    """Sum power around each fault frequency and its first few harmonics."""
    targets = {"bpfo": faults.bpfo_hz, "bpfi": faults.bpfi_hz, "bsf": faults.bsf_hz, "ftf": faults.ftf_hz}
    out = {}
    for name, base_hz in targets.items():
        energy = sum(
            _band_energy(freqs, power, base_hz * h, BAND_HALF_WIDTH_HZ * h)
            for h in range(1, N_HARMONICS + 1)
        )
        out[f"{prefix}_{name}_energy"] = energy
    return out


def frequency_domain_features(windows: np.ndarray, fs: float, faults: FaultFrequencies) -> dict[str, np.ndarray]:
    """Compute FFT- and envelope-spectrum features for each window.

    Parameters
    ----------
    windows : array of shape (n_windows, window_size)
    fs : sample rate in Hz
    faults : this bearing's BPFO/BPFI/BSF/FTF, from fault_frequencies.py

    Returns a dict of feature_name -> array of shape (n_windows,).
    """
    _, window_size = windows.shape
    freqs = np.fft.rfftfreq(window_size, d=1.0 / fs)

    rows = []
    for w in windows:
        w = w - w.mean()  # drop DC offset before either spectrum

        power = np.abs(np.fft.rfft(w)) ** 2
        total_energy = float(power.sum())
        dominant_hz = float(freqs[np.argmax(power[1:]) + 1])
        centroid_hz = float((freqs * power).sum() / total_energy) if total_energy > 0 else 0.0

        row = {
            "spectral_total_energy": total_energy,
            "dominant_frequency_hz": dominant_hz,
            "spectral_centroid_hz": centroid_hz,
        }
        row.update(_fault_band_energies(freqs, power, faults, prefix="raw"))

        envelope = np.abs(hilbert(w))
        envelope = envelope - envelope.mean()
        env_power = np.abs(np.fft.rfft(envelope)) ** 2
        row.update(_fault_band_energies(freqs, env_power, faults, prefix="env"))

        rows.append(row)

    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}