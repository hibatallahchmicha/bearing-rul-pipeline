"""Leak-safe train/test splits for RUL and health-index evaluation.

The single most common mistake in predictive-maintenance projects is
splitting *windows* rather than *units* (bearings). Because segmentation.py
uses 50% overlap, adjacent windows share half their samples -- a model
"tested" on a window that overlaps one it trained on isn't being tested at
all, it's recalling data it already saw. And even without overlap, windows
from later in a bearing's life correlate with windows from earlier in the
*same* bearing's life -- they share that bearing's specific noise floor,
mounting quirks, and resonance characteristics. A model can score well on
a held-out window from a bearing it already partially trained on without
having learned anything that generalizes to a bearing it's never seen.

This dataset only has a handful of bearings that run all the way to
failure (four, across all three test sets), so a single random 80/20
split would leave far too few units to trust the result. Leave-one-
bearing-out (LOBO) cross-validation is used instead: every bearing is held
out as the test unit exactly once, trained on the rest, results averaged
across folds -- more data-efficient, and honest about the small-n regime
this project is actually in.
"""

from __future__ import annotations

from collections.abc import Iterator


def leave_one_unit_out(unit_ids: list[str]) -> Iterator[tuple[list[str], str]]:
    """Yield (train_units, test_unit) folds, one per unit.

    Each unit (a bearing's run-to-failure trajectory) is held out exactly
    once. `train_units` never contains `test_unit` -- this is the only
    property that matters here, and it's asserted, not just implied.
    """
    if len(unit_ids) < 2:
        raise ValueError("need at least 2 units for leave-one-unit-out splitting")

    for test_unit in unit_ids:
        train_units = [u for u in unit_ids if u != test_unit]
        assert test_unit not in train_units
        yield train_units, test_unit