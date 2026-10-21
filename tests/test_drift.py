import numpy as np
import pytest

from bearing_pdm.drift import (
    PSI_DRIFT_THRESHOLD,
    detect_drift,
    population_stability_index,
    should_trigger_retrain,
)


def test_identical_distributions_have_near_zero_psi():
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)) < 0.02


def test_large_shift_has_high_psi():
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.normal(0, 1, 2000), rng.normal(2, 1, 2000)) > PSI_DRIFT_THRESHOLD


def test_large_samples_with_tiny_shift_are_not_drift():
    """The case that broke the p-value version: with thousands of samples a
    significance test flags a negligible shift. PSI measures size, so it
    does not."""
    rng = np.random.default_rng(0)
    psi = population_stability_index(rng.normal(0, 1, 5000), rng.normal(0.05, 1, 5000))
    assert psi < 0.1


def test_values_beyond_the_reference_range_still_register():
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.uniform(0, 1, 2000), rng.uniform(5, 6, 2000)) > PSI_DRIFT_THRESHOLD


def test_psi_rejects_empty_input():
    with pytest.raises(ValueError):
        population_stability_index(np.array([]), np.array([1.0]))


def test_stable_distribution_not_flagged():
    rng = np.random.default_rng(2)
    reference = {"a": rng.normal(0, 1, 3000), "b": rng.normal(0, 1, 3000)}
    current = {"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)}
    assert not detect_drift(reference, current).drift_detected


def test_shifted_distribution_flagged():
    rng = np.random.default_rng(2)
    reference = {"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300)}
    current = {"a": rng.normal(3, 1, 100), "b": rng.normal(3, 1, 100)}
    report = detect_drift(reference, current)
    assert report.drift_detected
    assert set(report.psi) == {"a", "b"}


def test_retrain_needs_consecutive_drift_reports():
    rng = np.random.default_rng(2)
    reference = {"a": rng.normal(0, 1, 300)}
    drifted_report = detect_drift(reference, {"a": rng.normal(5, 1, 100)})

    assert should_trigger_retrain([drifted_report] * 3)
    assert not should_trigger_retrain([drifted_report])