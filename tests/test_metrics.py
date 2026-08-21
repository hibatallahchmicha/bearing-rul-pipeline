import numpy as np
import pytest

from bearing_pdm.metrics import asymmetric_score, rul_metrics


def test_perfect_prediction_scores_zero():
    y = np.array([100.0, 50.0, 10.0])
    assert asymmetric_score(y, y) == pytest.approx(0.0)


def test_late_predictions_penalised_harder_than_early():
    """The defining property of the metric.

    Same error magnitude, opposite direction. Late means the model claimed
    more remaining life than the bearing had, so it fails in service. If
    this assertion ever inverts, every model-selection decision built on
    this score flips sign and nothing else in the suite notices.
    """
    y_true = np.array([50.0])
    early = np.array([30.0])
    late = np.array([70.0])
    assert asymmetric_score(y_true, late) > asymmetric_score(y_true, early)


def test_score_grows_with_error_magnitude():
    y_true = np.array([50.0])
    assert asymmetric_score(y_true, np.array([80.0])) > asymmetric_score(y_true, np.array([60.0]))


def test_score_stays_finite_on_an_absurd_prediction():
    assert np.isfinite(asymmetric_score(np.array([0.0]), np.array([1e9])))


def test_late_fraction_counts_only_over_predictions():
    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    y_pred = np.array([20.0, 20.0, 5.0, 10.0])  # 2 late, 1 early, 1 exact
    assert rul_metrics(y_true, y_pred).late_fraction == pytest.approx(0.5)


def test_mean_late_error_is_zero_when_never_late():
    assert rul_metrics(np.array([10.0, 10.0]), np.array([5.0, 8.0])).mean_late_error == 0.0


def test_mean_late_error_averages_only_the_late_ones():
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([16.0, 0.0])  # +6 late, -10 early
    assert rul_metrics(y_true, y_pred).mean_late_error == pytest.approx(6.0)


def test_mae_and_rmse_on_known_values():
    m = rul_metrics(np.array([0.0, 0.0]), np.array([3.0, -1.0]))
    assert m.mae == pytest.approx(2.0)
    assert m.rmse == pytest.approx(np.sqrt(5.0))


def test_n_is_carried_for_cross_fold_comparability():
    assert rul_metrics(np.zeros(7), np.zeros(7)).n == 7


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        rul_metrics(np.zeros(3), np.zeros(4))


def test_rejects_empty_prediction_set():
    with pytest.raises(ValueError):
        rul_metrics(np.array([]), np.array([]))