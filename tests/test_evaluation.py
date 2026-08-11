import pytest

from bearing_pdm.evaluation import leave_one_unit_out


def test_no_unit_appears_on_both_sides():
    units = ["1st_test_b3", "1st_test_b4", "2nd_test_b1", "3rd_test_b3"]
    for train_units, test_unit in leave_one_unit_out(units):
        assert test_unit not in train_units


def test_every_unit_held_out_exactly_once():
    units = ["a", "b", "c", "d"]
    test_units_seen = [test_unit for _, test_unit in leave_one_unit_out(units)]
    assert sorted(test_units_seen) == sorted(units)


def test_requires_at_least_two_units():
    with pytest.raises(ValueError):
        list(leave_one_unit_out(["only_one"]))
