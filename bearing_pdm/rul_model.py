"""Degradation-stage estimation: healthy / degrading / critical.

Stage classification, not RUL regression, is the primary output here.
With only 4 bearings in the whole dataset observed running all the way to
failure, a regressor learning "trajectory shape -> time-to-failure" would
effectively be fit on 4 independent examples of what failure looks like --
leave-one-bearing-out CV (evaluation.py) can measure that honestly, but
can't manufacture more real trajectories. Stage classification only needs
to learn thresholds on the health index, and each of those 4 bearings
still contributes thousands of self-labeled windows (labeled by their own
position in that bearing's life), which is a far better-conditioned
problem at this sample size. RUL regression stays a documented option for
later, deliberately not built now.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fraction of a bearing's remaining life at which it crosses into the next
# stage. Honest defaults, not tuned values -- see CHECKPOINTS.md.
DEGRADING_AT_FRACTION_REMAINING = 0.30
CRITICAL_AT_FRACTION_REMAINING = 0.10


def fraction_of_life_remaining(n_windows: int) -> np.ndarray:
    """Ground-truth label for a bearing's OWN complete run-to-failure life:
    1.0 at the first window, 0.0 at the last (failure) window. Only valid
    for bearings observed running all the way to failure.
    """
    idx = np.arange(n_windows)
    return 1.0 - idx / (n_windows - 1)


def true_stage(fraction_remaining: np.ndarray) -> np.ndarray:
    """Ground-truth stage label from fraction-of-life-remaining."""
    stage = np.full(fraction_remaining.shape, "healthy", dtype=object)
    stage[fraction_remaining <= DEGRADING_AT_FRACTION_REMAINING] = "degrading"
    stage[fraction_remaining <= CRITICAL_AT_FRACTION_REMAINING] = "critical"
    return stage


@dataclass(frozen=True)
class StageThresholds:
    degrading_at: float
    critical_at: float


def fit_stage_thresholds(
    train_health_scores: list[np.ndarray], train_fraction_remaining: list[np.ndarray]
) -> StageThresholds:
    """Learn health-index thresholds for degrading/critical from labeled
    training bearings (the fold's train_units only -- never the held-out
    test bearing).

    Threshold = the health-index score typically seen right at the true
    stage boundary, across training bearings.
    """
    all_scores = np.concatenate(train_health_scores)
    all_stage = true_stage(np.concatenate(train_fraction_remaining))

    return StageThresholds(
        degrading_at=float(np.median(all_scores[all_stage == "degrading"])),
        critical_at=float(np.median(all_scores[all_stage == "critical"])),
    )


def predict_stage(health_scores: np.ndarray, thresholds: StageThresholds) -> np.ndarray:
    """Classify windows into healthy/degrading/critical from health-index score."""
    stage = np.full(health_scores.shape, "healthy", dtype=object)
    stage[health_scores >= thresholds.degrading_at] = "degrading"
    stage[health_scores >= thresholds.critical_at] = "critical"
    return stage