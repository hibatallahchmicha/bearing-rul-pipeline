import numpy as np
import pytest

from bearing_pdm.normalization import normalize_to_own_baseline


def _bearing(n=100, n_features=3, seed=0):
    return np.random.default_rng(seed).normal(5.0, 2.0, size=(n, n_features))


def test_baseline_period_becomes_zero_mean_unit_std():
    out = normalize_to_own_baseline(_bearing(), healthy_fraction=0.2)
    baseline = out[:20]
    assert np.allclose(baseline.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(baseline.std(axis=0), 1.0, atol=1e-9)


def test_late_life_rows_never_influence_the_scaling():
    """The leakage guard: changing the failing end of a bearing's life must
    not change how its healthy period is normalized."""
    original = _bearing()
    altered = original.copy()
    altered[50:] *= 100.0  # a dramatic "failure" in the second half

    a = normalize_to_own_baseline(original, healthy_fraction=0.2)
    b = normalize_to_own_baseline(altered, healthy_fraction=0.2)
    assert np.allclose(a[:50], b[:50])


def test_constant_feature_does_not_divide_by_zero():
    matrix = _bearing()
    matrix[:, 1] = 7.0
    out = normalize_to_own_baseline(matrix)
    assert np.all(np.isfinite(out))
    assert np.allclose(out[:, 1], 0.0)


def test_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        normalize_to_own_baseline(_bearing(), healthy_fraction=0.0)


def test_rejects_one_dimensional_input():
    with pytest.raises(ValueError):
        normalize_to_own_baseline(np.arange(10.0))