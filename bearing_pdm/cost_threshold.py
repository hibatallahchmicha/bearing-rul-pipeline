"""Cost-sensitive decision threshold for the critical-stage alarm.

Module 8's `critical_at` threshold is a *typical* value (the median
health-index score at the true critical boundary in training data) --
not a value chosen because it minimizes real-world cost. A maintenance
alarm system has two distinct error types with very different
consequences: a false alarm costs a wasted inspection; a missed failure
costs an unplanned breakdown -- lost production, an emergency repair
premium, and for the worst cases, collateral damage. Treating those as
equally costly, which any threshold chosen by accuracy or "typical value"
implicitly does, is not a decision a real maintenance operation accepts.

This module sweeps candidate thresholds on the health index and picks the
one minimizing *expected total cost*, given a stated ratio between the
two error costs, rather than a threshold that merely looks statistically
typical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Illustrative cost ratio, not sourced from a real maintenance budget: an
# unplanned failure is assumed to cost 10x a wasted inspection. Swap these
# for real figures if they ever exist -- everything downstream depends
# only on the ratio between them, not their units.
COST_FALSE_ALARM = 1.0
COST_MISSED_FAILURE = 10.0


@dataclass(frozen=True)
class ThresholdSweepResult:
    threshold: float
    expected_cost: float
    false_alarm_rate: float
    missed_failure_rate: float


def sweep_thresholds(
    health_scores: np.ndarray,
    is_critical: np.ndarray,
    n_candidates: int = 100,
    cost_false_alarm: float = COST_FALSE_ALARM,
    cost_missed_failure: float = COST_MISSED_FAILURE,
) -> list[ThresholdSweepResult]:
    """Evaluate expected cost across a range of candidate thresholds.

    `is_critical` is the ground-truth boolean label for each window in
    `health_scores`. Both must come from the fold's TRAINING bearings --
    never from the held-out unit. Sweeping on the unit you then report
    against selects the threshold that happens to suit that unit's own
    labels, which is leakage: it reports a threshold's best-case
    performance as if it were its expected performance. An earlier
    version of run_pipeline.py did exactly this, and fixing it raised the
    measured missed-failure rate on one fold from 0.005 to 0.771.

    Note also that expected cost is computed from raw COUNTS, not rates,
    so pooling training bearings of unequal length lets the longest unit
    dominate the chosen threshold.
    """
    candidates = np.linspace(health_scores.min(), health_scores.max(), n_candidates)
    results = []
    for t in candidates:
        predicted_critical = health_scores >= t
        false_alarms = np.sum(predicted_critical & ~is_critical)
        missed_failures = np.sum(~predicted_critical & is_critical)

        results.append(ThresholdSweepResult(
            threshold=float(t),
            expected_cost=float(false_alarms * cost_false_alarm + missed_failures * cost_missed_failure),
            false_alarm_rate=float(false_alarms / max(np.sum(~is_critical), 1)),
            missed_failure_rate=float(missed_failures / max(np.sum(is_critical), 1)),
        ))
    return results


def best_threshold(results: list[ThresholdSweepResult]) -> ThresholdSweepResult:
    """Pick the threshold minimizing expected cost."""
    return min(results, key=lambda r: r.expected_cost)