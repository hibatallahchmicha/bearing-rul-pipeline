import numpy as np

from bearing_pdm.rul_model import fit_stage_thresholds, fraction_of_life_remaining, predict_stage, true_stage


def _fake_bearing(rng, n=200, slope=30.0):
    fraction = fraction_of_life_remaining(n)
    return 1.0 + (1 - fraction) * slope + rng.normal(0, 1, n), fraction


def test_critical_threshold_above_degrading_threshold():
    rng = np.random.default_rng(0)
    scores_a, frac_a = _fake_bearing(rng)
    scores_b, frac_b = _fake_bearing(rng)
    thresholds = fit_stage_thresholds([scores_a, scores_b], [frac_a, frac_b])
    assert thresholds.critical_at > thresholds.degrading_at


def test_held_out_bearing_beats_random_guessing():
    rng = np.random.default_rng(0)
    scores_a, frac_a = _fake_bearing(rng)
    scores_b, frac_b = _fake_bearing(rng)
    thresholds = fit_stage_thresholds([scores_a, scores_b], [frac_a, frac_b])

    scores_c, frac_c = _fake_bearing(rng)
    predicted = predict_stage(scores_c, thresholds)
    actual = true_stage(frac_c)
    assert (predicted == actual).mean() > 0.33  # beats 3-class random guessing
