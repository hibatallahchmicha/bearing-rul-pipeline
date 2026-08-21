"""MLflow run logging for leave-one-bearing-out evaluation.

Kept deliberately thin. This module only records what happened; it does
not decide anything, and it holds no model logic.

**Local file store by default.** `mlruns/` in the repo root works with no
account, no cluster and no network, which keeps the test suite and CI
hermetic. Moving to Databricks is a one-line change of tracking URI --
the logging calls are identical -- so nothing here has to be rewritten
when the platform work lands.

**Nested runs, one parent per experiment and one child per fold.** The
alternative -- a single flat run holding averaged metrics -- would hide
exactly the thing that matters most in this project: fold-to-fold
variance. With four units, the spread between folds carries more
information than the mean does, and a flat run makes that spread
invisible in the UI.

**Per-prediction means, not sums, for cross-fold aggregates.** The
asymmetric score is a sum, so folds of unequal length are not directly
comparable: `3rd_test_b3` contributes ~633 snapshots against
`2nd_test_b1`'s ~99. Averaging raw sums would let the longest bearing
dominate the headline number, so aggregates divide by `n` first.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import mlflow

from bearing_pdm.metrics import RULMetrics

DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_EXPERIMENT = "bearing-rul-lobo"


def configure_tracking(
    tracking_uri: str = DEFAULT_TRACKING_URI, experiment: str = DEFAULT_EXPERIMENT
) -> None:
    """Point MLflow at a tracking store and experiment.

    On Databricks, pass tracking_uri="databricks" and an experiment path
    such as "/Users/<you>/bearing-rul-lobo". Nothing else changes.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)


@contextmanager
def parent_run(run_name: str, params: Mapping[str, Any]) -> Iterator[Any]:
    """Open the top-level run for one predictor's full LOBO evaluation."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(dict(params))
        yield run


@contextmanager
def fold_run(test_unit: str) -> Iterator[Any]:
    """Open a nested child run for a single held-out bearing."""
    with mlflow.start_run(run_name=f"fold_{test_unit}", nested=True) as run:
        mlflow.log_param("test_unit", test_unit)
        yield run


def log_rul_metrics(metrics: RULMetrics, prefix: str = "") -> None:
    """Log one RULMetrics object.

    `asymmetric_score_per_prediction` is logged alongside the raw sum
    precisely so that cross-fold comparison uses the length-independent
    quantity by default.
    """
    mlflow.log_metrics(
        {
            f"{prefix}asymmetric_score": metrics.asymmetric_score,
            f"{prefix}asymmetric_score_per_prediction": metrics.asymmetric_score / metrics.n,
            f"{prefix}late_fraction": metrics.late_fraction,
            f"{prefix}mean_late_error_hours": metrics.mean_late_error,
            f"{prefix}mae_hours": metrics.mae,
            f"{prefix}rmse_hours": metrics.rmse,
            f"{prefix}n_predictions": float(metrics.n),
        }
    )