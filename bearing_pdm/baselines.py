"""Naive RUL baselines: the floor a real model has to clear.

Reporting that a model achieves some RMSE means nothing without saying
what the same metric looks like for a predictor that knows nothing. These
two baselines run on the identical leave-one-bearing-out split as the
model, through the identical metric code, so the comparison is structural
rather than rhetorical.

**Why not a literal "last-value / persistence" baseline.** In forecasting,
persistence predicts that the next value equals the last observed one. RUL
has no equivalent at inference time: knowing remaining life at the previous
timestep would require a label you do not have in deployment. A persistence
baseline for RUL is an oracle, so it is not built here.

**MeanRULBaseline** is the information floor -- one constant, no features.
Failing to beat it means the sensors contributed nothing.

**LinearDegradationBaseline** is the adversarial one. Elapsed operating
time IS known in deployment; only total lifespan is unknown. So "the
average bearing lasts L hours, this one has run t, therefore L-t remain"
is a legitimate deployable predictor that uses no sensor data at all. If
FFT- and envelope-derived features cannot beat a clock, the entire
signal-processing stage has no measured value -- which is precisely the
"is this just curve-fitting a monotonic trend?" question that any serious
reviewer will ask about a degradation model.
"""

from __future__ import annotations

import numpy as np


class MeanRULBaseline:
    """Predict the training set's mean capped RUL for every input.

    Uses no features whatsoever. This is the "no information" floor.
    """

    def __init__(self) -> None:
        self.value_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MeanRULBaseline:
        if len(y) == 0:
            raise ValueError("cannot fit a baseline on an empty training set")
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.value_ is None:
            raise RuntimeError("MeanRULBaseline.predict called before fit")
        return np.full(len(X), self.value_, dtype=float)


class LinearDegradationBaseline:
    """Fit RUL as a straight line in elapsed operating hours, ignoring sensors.

    `elapsed_col` is the index of the elapsed-operating-hours column in the
    feature matrix. Every other column is deliberately discarded: the point
    of this baseline is to measure what a clock alone achieves, so that the
    model's margin over it is the measured value of the vibration features.

    Predictions are clipped at zero -- a negative remaining life is not a
    meaningful output, and letting the line run below zero would flatter
    the asymmetric score by producing impossibly early predictions.
    """

    def __init__(self, elapsed_col: int) -> None:
        self.elapsed_col = elapsed_col
        self.slope_: float | None = None
        self.intercept_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearDegradationBaseline:
        if len(y) == 0:
            raise ValueError("cannot fit a baseline on an empty training set")
        elapsed = np.asarray(X)[:, self.elapsed_col].astype(float)
        self.slope_, self.intercept_ = np.polyfit(elapsed, np.asarray(y, dtype=float), deg=1)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.slope_ is None or self.intercept_ is None:
            raise RuntimeError("LinearDegradationBaseline.predict called before fit")
        elapsed = np.asarray(X)[:, self.elapsed_col].astype(float)
        return np.clip(self.slope_ * elapsed + self.intercept_, 0.0, None)