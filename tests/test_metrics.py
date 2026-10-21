import numpy as np
import pytest

from bearing_pdm.metrics import (
    DEFAULT_ALERT_HORIZON_HOURS,
    alarm_cost,
    asymmetric_score,
    provable_false_alarm_rate,
    rul_metrics,
)


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

def test_alert_inside_the_horizon_is_not_a_false_alarm():
    hours_to_end = np.array([100.0, 30.0])
    predicted = np.array([150.0, 10.0])  # alerts only at 30 h before end
    assert provable_false_alarm_rate(predicted, hours_to_end) == 0.0


def test_alert_far_from_the_end_is_a_false_alarm():
    hours_to_end = np.array([200.0, 100.0, 30.0])
    predicted = np.array([150.0, 20.0, 10.0])  # alert at 100 h: provably false
    assert provable_false_alarm_rate(predicted, hours_to_end) == 0.5


def test_no_judgeable_snapshots_returns_nan_so_gates_fail_safe():
    rate = provable_false_alarm_rate(np.array([5.0, 3.0]), np.array([20.0, 10.0]))
    assert np.isnan(rate)
    assert not (rate <= 0.05)


def test_default_horizon_is_48_hours():
    assert DEFAULT_ALERT_HORIZON_HOURS == 48.0


def test_false_alarm_rate_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        provable_false_alarm_rate(np.array([1.0, 2.0]), np.array([1.0]))

def test_alarm_cost_counts_both_mistakes():
    hours_to_end = np.array([100.0, 100.0, 20.0, 20.0])
    predicted = np.array([10.0, 150.0, 10.0, 150.0])
    # row 0: alert, 100 h left  -> false alarm
    # row 1: silent, 100 h left -> correct
    # row 2: alert, 20 h left   -> correct
    # row 3: silent, 20 h left  -> missed alarm
    result = alarm_cost(predicted, hours_to_end, censored=np.zeros(4, dtype=bool))
    assert result.false_alarms == 1
    assert result.missed_alarms == 1


def test_missed_alarm_costs_ten_times_a_false_alarm_by_default():
    one_false = alarm_cost(np.array([10.0]), np.array([100.0]), censored=np.array([False]))
    one_missed = alarm_cost(np.array([150.0]), np.array([20.0]), censored=np.array([False]))
    assert one_missed.cost == 10 * one_false.cost


def test_censored_bearing_never_counts_as_missed():
    """Silence near the end of a healthy bearing's recording can't be proven
    wrong -- it was never seen failing."""
    result = alarm_cost(np.array([150.0]), np.array([20.0]), censored=np.array([True]))
    assert result.missed_alarms == 0
    assert result.cost == 0


def test_censored_bearing_still_counts_provable_false_alarms():
    result = alarm_cost(np.array([10.0]), np.array([100.0]), censored=np.array([True]))
    assert result.false_alarms == 1


def test_alarm_cost_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        alarm_cost(np.array([1.0, 2.0]), np.array([1.0]), censored=np.array([False]))