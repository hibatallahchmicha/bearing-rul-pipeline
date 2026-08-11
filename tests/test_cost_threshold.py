import numpy as np

from bearing_pdm.cost_threshold import best_threshold, sweep_thresholds


def test_higher_missed_failure_cost_lowers_threshold():
    rng = np.random.default_rng(1)
    n = 500
    is_critical = rng.random(n) < 0.15
    scores = np.where(is_critical, rng.normal(30, 3, n), rng.normal(15, 3, n))

    default = best_threshold(sweep_thresholds(scores, is_critical))
    strict = best_threshold(sweep_thresholds(scores, is_critical, cost_missed_failure=100.0))

    assert strict.threshold < default.threshold
    assert strict.missed_failure_rate <= default.missed_failure_rate
