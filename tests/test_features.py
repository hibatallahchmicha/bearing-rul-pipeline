import numpy as np

from bearing_pdm.fault_frequencies import compute_fault_frequencies
from bearing_pdm.features_freq import frequency_domain_features
from bearing_pdm.features_time import time_domain_features


def test_time_domain_features_shape():
    windows = np.random.default_rng(0).normal(size=(5, 2048))
    feats = time_domain_features(windows)
    assert set(feats) == {"rms", "std", "kurtosis", "skewness", "crest_factor", "peak_to_peak"}
    assert all(v.shape == (5,) for v in feats.values())


def test_frequency_features_identify_correct_fault_band():
    fs = 20000
    faults = compute_fault_frequencies(shaft_speed_rpm=2000)
    t = np.arange(2048) / fs
    windows = np.sin(2 * np.pi * faults.bpfo_hz * t)[np.newaxis, :]

    feats = frequency_domain_features(windows, fs=fs, faults=faults)
    assert feats["raw_bpfo_energy"][0] > feats["raw_bpfi_energy"][0]
    assert feats["raw_bpfo_energy"][0] > feats["raw_bsf_energy"][0]
    assert feats["raw_bpfo_energy"][0] > feats["raw_ftf_energy"][0]
