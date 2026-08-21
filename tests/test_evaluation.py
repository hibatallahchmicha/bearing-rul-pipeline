import numpy as np
import pytest

from bearing_pdm.evaluation import assert_no_unit_overlap, leave_one_unit_out, split_by_unit

UNITS = ["1st_test_b3", "1st_test_b4", "2nd_test_b1", "3rd_test_b3"]


def test_no_unit_appears_on_both_sides():
    for train_units, test_unit in leave_one_unit_out(UNITS):
        assert test_unit not in train_units


def test_every_unit_held_out_exactly_once():
    units = ["a", "b", "c", "d"]
    test_units_seen = [test_unit for _, test_unit in leave_one_unit_out(units)]
    assert sorted(test_units_seen) == sorted(units)


def test_requires_at_least_two_units():
    with pytest.raises(ValueError):
        list(leave_one_unit_out(["only_one"]))


def _row_labels() -> np.ndarray:
    """Three units with deliberately unequal row counts, as in the real data."""
    return np.array(["b3"] * 5 + ["b4"] * 3 + ["b1"] * 4)


def test_split_by_unit_masks_are_disjoint_and_complete():
    labels = _row_labels()
    for train_mask, test_mask in split_by_unit(labels):
        assert not np.any(train_mask & test_mask)
        assert np.all(train_mask | test_mask)


def test_split_by_unit_holds_out_whole_units():
    labels = _row_labels()
    for train_mask, test_mask in split_by_unit(labels):
        train_units = set(labels[train_mask])
        test_units = set(labels[test_mask])
        assert len(test_units) == 1
        assert not (train_units & test_units)


def test_split_by_unit_covers_every_unit_once():
    labels = _row_labels()
    held_out = [set(labels[test_mask]).pop() for _, test_mask in split_by_unit(labels)]
    assert sorted(held_out) == sorted(set(labels.tolist()))


def test_split_by_unit_requires_two_distinct_units():
    with pytest.raises(ValueError):
        list(split_by_unit(np.array(["b3"] * 10)))


def test_detects_reintroduced_row_level_leakage():
    """The regression test: catch a naive positional split.

    This is the mistake someone makes when they have a flat feature table
    and reach for `df[:80]` / `df[80:]` without thinking about units. It
    splits mid-unit, so the same bearing lands on both sides.
    """
    labels = _row_labels()
    n = len(labels)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[: n // 2] = True
    test_mask = ~train_mask
    with pytest.raises(AssertionError, match="appear in BOTH train and test"):
        assert_no_unit_overlap(labels, train_mask, test_mask)


def test_detects_overlapping_masks():
    labels = _row_labels()
    everything = np.ones(len(labels), dtype=bool)
    with pytest.raises(AssertionError, match="BOTH train and test masks"):
        assert_no_unit_overlap(labels, everything, everything)


def test_rejects_empty_partition():
    labels = _row_labels()
    all_train = np.ones(len(labels), dtype=bool)
    none_test = np.zeros(len(labels), dtype=bool)
    with pytest.raises(AssertionError, match="test partition is empty"):
        assert_no_unit_overlap(labels, all_train, none_test)


def test_rejects_mask_length_mismatch():
    labels = _row_labels()
    with pytest.raises(ValueError, match="mask length mismatch"):
        assert_no_unit_overlap(labels, np.ones(3, dtype=bool), np.zeros(3, dtype=bool))
