"""RUL regression over the full feature set, with train-only scaling.

Interchangeable with the naive predictors in `baselines.py` -- same
`fit`/`predict` interface, same split, same metrics -- so the comparison
between "uses the vibration features" and "uses a clock" is apples to
apples.

**Ridge as the default.** With only four run-to-failure units, a
high-capacity learner will memorise per-bearing idiosyncrasies rather than
degradation physics; that failure mode already broke the alarm threshold's
cross-bearing transfer in this project, so it is a demonstrated risk here
and not a hypothetical one. Ridge's L2 penalty is the appropriate prior at
this sample size, and its coefficients are directly interpretable -- which
matters when the audience is a maintenance engineer rather than a data
scientist.

**Gradient boosting as the challenger, not as a rejected option.** It may
well win. Both are logged as MLflow runs on the identical split, so the
experiment record decides rather than this docstring. Picking a model
because of a plausible argument about sample size, without measuring the
alternative, is the thing experiment tracking exists to stop.

**Scaling is fitted inside each fold, on training units only.** Ridge is
scale-sensitive and these features span roughly six orders of magnitude
(spectral energies are O(1e5); kurtosis is O(1)). Fitting a scaler on all
data before splitting would leak held-out distribution statistics into
training -- a quieter relative of the two leaks already found and fixed in
this project, and the one most often missed because nothing fails loudly
when you get it wrong.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 3
RANDOM_STATE = 42


def build_ridge_regressor(alpha: float = DEFAULT_RIDGE_ALPHA) -> Pipeline:
    """Ridge regression with scaling fitted inside the pipeline.

    Wrapping the scaler in a Pipeline is what makes the train-only
    guarantee structural: `pipeline.fit(X_train, y_train)` fits the
    scaler on training rows alone, and `pipeline.predict(X_test)` reuses
    those fitted statistics. There is no code path in which test data
    can reach `scaler.fit`.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=None)),
        ]
    )


def build_gradient_boosting_regressor(
    n_estimators: int = DEFAULT_N_ESTIMATORS, max_depth: int = DEFAULT_MAX_DEPTH
) -> Pipeline:
    """Gradient-boosted trees. Scaling is redundant for trees but kept so
    both candidates share one interface and one code path.

    `max_depth=3` is a deliberate capacity limit rather than a default:
    deeper trees at four units would fit bearing identity rather than
    degradation.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def predict_clipped(model: Pipeline, X: np.ndarray, cap_hours: float) -> np.ndarray:
    """Predict, then clamp into the target's valid range [0, cap].

    The target is capped by construction (see rul_target.py), so a
    prediction outside that range is unphysical regardless of what the
    regressor produced. Clipping is applied at predict time rather than
    baked into the model so that the raw model stays a standard sklearn
    estimator -- which keeps MLflow's signature inference and any future
    serving path straightforward.
    """
    return np.clip(model.predict(X), 0.0, cap_hours)