"""Leak-safe train/test splits for health-index and degradation-stage evaluation.

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
split would leave far too few units to trust the result. Leave-one-bearing-out (LOBO)
cross-validation is used instead: every bearing is held out as the test unit exactly once,
trained on the rest, results averaged across folds -- more data-efficient, and honest about
the small-n regime this project is actually in.

Two APIs are provided at different levels:
- `leave_one_unit_out` works on a list of unit IDs. The caller then selects its own data
  for those units.
- `split_by_unit` works on a flat, row-aligned array of unit labels -- one label per feature
  row -- and returns boolean row masks directly. This is the API to use once features live in
  a single table, because it removes the caller's indexing step, which is precisely where the
  leak gets reintroduced in practice.
Both route their guarantee through `assert_no_unit_overlap`, which raises an explicit
`AssertionError` rather than using a bare `assert` statement. This is deliberate: Python removes
every `assert` under `-O` / `PYTHONOPTIMIZE`, so a leakage guard written as a bare assert
silently disappears in optimized execution -- exactly the environment a scheduled job is most
likely to run in.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np


def assert_no_unit_overlap(
    unit_labels: Sequence[str] | np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    """Raise if any unit, or any row, appears on both sides of a split.

    Checks three independent properties, because each fails differently:
    1. No unit ID appears in both partitions -- the leakage this whole
       module exists to prevent.
    2. No row is selected by both masks -- catches a caller that built
       masks by hand and got the logic wrong.
    3. Neither partition is empty -- an empty test set makes any metric
       computed from it meaningless rather than merely wrong.
    Raises `AssertionError` explicitly rather than using a bare `assert`,
    so the guarantee survives `python -O`.
    """
    labels = np.asarray(unit_labels)
    if labels.shape[0] != train_mask.shape[0] or labels.shape[0] != test_mask.shape[0]:
        raise ValueError(
            f"mask length mismatch: {labels.shape[0]} labels, "
            f"{train_mask.shape[0]} train mask, {test_mask.shape[0]} test mask"
        )
    if bool(np.any(train_mask & test_mask)):
        n = int(np.sum(train_mask & test_mask))
        raise AssertionError(f"{n} row(s) selected by BOTH train and test masks")

    train_units = set(labels[train_mask].tolist())
    test_units = set(labels[test_mask].tolist())
    overlap = train_units & test_units
    if overlap:
        raise AssertionError(
            f"unit(s) {sorted(overlap)} appear in BOTH train and test -- "
            "this is the leakage leave-one-bearing-out exists to prevent"
        )
    if not train_units:
        raise AssertionError("train partition is empty")
    if not test_units:
        raise AssertionError("test partition is empty")


def leave_one_unit_out(unit_ids: list[str]) -> Iterator[tuple[list[str], str]]:
    """Yield (train_units, test_unit) folds, one per unit.

    Each unit (a bearing's run-to-failure trajectory) is held out exactly
    once. `train_units` never contains `test_unit`.
    """
    if len(unit_ids) < 2:
        raise ValueError("need at least 2 units for leave-one-unit-out splitting")

    for test_unit in unit_ids:
        train_units = [u for u in unit_ids if u != test_unit]
        if test_unit in train_units:
            raise AssertionError(f"unit {test_unit!r} present in its own training set")
        yield train_units, test_unit


def split_by_unit(
    unit_labels: Sequence[str] | np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-unit-out over a flat, row-aligned array of unit labels.

    `unit_labels` has one entry per feature row -- the `unit_id` column of
    a model-ready feature table. Yields `(train_mask, test_mask)` boolean
    arrays over those rows, so the caller never performs the row selection
    itself.
    Unit order follows first appearance, so fold order is deterministic
    and reproducible across runs without depending on set iteration or
    dict hashing.
    """
    labels = np.asarray(unit_labels)
    if labels.ndim != 1:
        raise ValueError(f"expected a 1-D array of per-row unit labels, got shape {labels.shape}")
    unique_units = list(dict.fromkeys(labels.tolist()))
    if len(unique_units) < 2:
        raise ValueError(
            f"need at least 2 distinct units for leave-one-unit-out splitting, got {len(unique_units)}"
        )

    for test_unit in unique_units:
        test_mask = labels == test_unit
        train_mask = ~test_mask
        assert_no_unit_overlap(labels, train_mask, test_mask)
        yield train_mask, test_mask