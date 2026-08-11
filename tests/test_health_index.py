import numpy as np

from bearing_pdm.health_index import fit_healthy_baseline, mahalanobis_health_index, stack_features


def test_healthy_scores_low_degraded_scores_high():
    rng = np.random.default_rng(42)
    healthy = {"a": rng.normal(10, 1, 200), "b": rng.normal(5, 0.5, 200), "c": rng.normal(100, 10, 200)}
    matrix, _ = stack_features(healthy)
    mean, inv_cov = fit_healthy_baseline(matrix)

    healthy_scores = mahalanobis_health_index(matrix, mean, inv_cov)
    degraded_score = mahalanobis_health_index(np.array([[20.0, 9.0, 300.0]]), mean, inv_cov)[0]

    assert healthy_scores.mean() < 5
    assert degraded_score > healthy_scores.mean() * 5
