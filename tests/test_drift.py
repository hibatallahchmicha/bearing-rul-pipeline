import numpy as np

from bearing_pdm.drift import detect_drift, should_trigger_retrain


def test_stable_distribution_not_flagged():
    rng = np.random.default_rng(2)
    reference = {"a": rng.normal(0, 1, 300)}
    current = {"a": rng.normal(0, 1, 100)}
    assert not detect_drift(reference, current).drift_detected


def test_shifted_distribution_flagged():
    rng = np.random.default_rng(2)
    reference = {"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300)}
    current = {"a": rng.normal(3, 1, 100), "b": rng.normal(3, 1, 100)}
    assert detect_drift(reference, current).drift_detected


def test_retrain_needs_consecutive_drift_reports():
    rng = np.random.default_rng(2)
    reference = {"a": rng.normal(0, 1, 300)}
    drifted_report = detect_drift(reference, {"a": rng.normal(5, 1, 100)})

    assert should_trigger_retrain([drifted_report] * 3)
    assert not should_trigger_retrain([drifted_report])
