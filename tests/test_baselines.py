import numpy as np
import pytest

from bearing_pdm.baselines import LinearDegradationBaseline, MeanRULBaseline

ELAPSED_COL = 0


def test_mean_baseline_predicts_the_training_mean():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    preds = MeanRULBaseline().fit(np.zeros((4, 3)), y).predict(np.zeros((2, 3)))
    assert np.allclose(preds, 25.0)


def test_mean_baseline_ignores_features_entirely():
    model = MeanRULBaseline().fit(np.zeros((2, 3)), np.array([10.0, 20.0]))
    assert np.allclose(model.predict(np.zeros((5, 3))), model.predict(np.full((5, 3), 1e6)))


def test_mean_baseline_rejects_predict_before_fit():
    with pytest.raises(RuntimeError):
        MeanRULBaseline().predict(np.zeros((2, 3)))


def test_linear_baseline_recovers_a_known_relationship():
    elapsed = np.arange(0.0, 100.0)
    preds = (
        LinearDegradationBaseline(ELAPSED_COL)
        .fit(elapsed.reshape(-1, 1), 100.0 - elapsed)
        .predict(np.array([[25.0]]))
    )
    assert preds[0] == pytest.approx(75.0, abs=1e-6)


def test_linear_baseline_uses_only_the_elapsed_column():
    """This baseline exists to answer 'do the FFT features beat a clock?'

    If it ever starts reading sensor columns it stops being a baseline and
    the comparison silently becomes meaningless -- with no failure anywhere.
    """
    elapsed = np.arange(0.0, 50.0)
    noise = np.random.default_rng(0).normal(size=50)
    model = LinearDegradationBaseline(ELAPSED_COL).fit(
        np.column_stack([elapsed, noise]), 50.0 - elapsed
    )
    quiet = model.predict(np.array([[10.0, 0.0]]))
    noisy = model.predict(np.array([[10.0, 1e6]]))
    assert quiet == pytest.approx(noisy)


def test_linear_baseline_never_predicts_negative_life():
    elapsed = np.arange(0.0, 100.0)
    model = LinearDegradationBaseline(ELAPSED_COL).fit(
        elapsed.reshape(-1, 1), np.clip(100.0 - elapsed, 0.0, None)
    )
    assert model.predict(np.array([[500.0]]))[0] >= 0.0


def test_linear_baseline_rejects_predict_before_fit():
    with pytest.raises(RuntimeError):
        LinearDegradationBaseline(ELAPSED_COL).predict(np.zeros((2, 2)))


def test_both_baselines_reject_an_empty_training_set():
    for model in (MeanRULBaseline(), LinearDegradationBaseline(ELAPSED_COL)):
        with pytest.raises(ValueError):
            model.fit(np.zeros((0, 2)), np.array([]))