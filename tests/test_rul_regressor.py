import numpy as np

from bearing_pdm.rul_regressor import (
    build_gradient_boosting_regressor,
    build_ridge_regressor,
    predict_clipped,
)


def _linear_problem(n: int = 200, n_features: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = np.clip(100.0 - 20.0 * X[:, 0], 0.0, 168.0)
    return X, y


def test_ridge_beats_a_constant_predictor_on_a_learnable_signal():
    X, y = _linear_problem()
    model = build_ridge_regressor().fit(X, y)
    assert np.mean(np.abs(model.predict(X) - y)) < np.mean(np.abs(y - y.mean()))


def test_gradient_boosting_builds_and_fits():
    X, y = _linear_problem()
    model = build_gradient_boosting_regressor(n_estimators=20).fit(X, y)
    assert model.predict(X).shape == y.shape


def test_scaler_is_fitted_on_training_rows_only():
    """The quietest leak class: a scaler that saw held-out data.

    Nothing fails when you get this wrong -- the metrics just improve.
    Wrapping the scaler in a Pipeline makes the guarantee structural, and
    this test pins that guarantee down.
    """
    model = build_ridge_regressor().fit(np.zeros((10, 3)), np.full(10, 50.0))
    assert np.allclose(model.named_steps["scaler"].mean_, 0.0)

    model.predict(np.full((10, 3), 1000.0))
    assert np.allclose(model.named_steps["scaler"].mean_, 0.0)


def test_predict_clipped_bounds_output_to_the_target_range():
    X, y = _linear_problem()
    model = build_ridge_regressor().fit(X, y)
    preds = predict_clipped(model, X * 50.0, cap_hours=168.0)
    assert preds.min() >= 0.0
    assert preds.max() <= 168.0