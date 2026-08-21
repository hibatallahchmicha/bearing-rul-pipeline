from datetime import datetime, timedelta

import numpy as np
import pytest

from bearing_pdm.rul_target import (
    DEFAULT_RUL_CAP_HOURS,
    aggregate_windows_to_snapshot,
    build_rul_target,
    cap_rul,
    rul_hours_from_timestamps,
)


def _timestamps(n: int, hours_apart: float = 1.0) -> list[datetime]:
    start = datetime(2003, 10, 22, 12, 0, 0)
    return [start + timedelta(hours=i * hours_apart) for i in range(n)]


def test_final_timestamp_has_zero_rul():
    assert rul_hours_from_timestamps(_timestamps(5))[-1] == 0.0


def test_rul_decreases_monotonically():
    rul = rul_hours_from_timestamps(_timestamps(10))
    assert np.all(np.diff(rul) < 0)


def test_rul_is_measured_in_hours():
    ts = _timestamps(3, hours_apart=2.0)
    assert rul_hours_from_timestamps(ts)[0] == pytest.approx(4.0)


def test_requires_at_least_two_timestamps():
    with pytest.raises(ValueError):
        rul_hours_from_timestamps(_timestamps(1))


def test_rejects_non_increasing_timestamps():
    ts = _timestamps(4)
    ts[1] = ts[2]  # duplicate -> ordering broke upstream
    with pytest.raises(ValueError, match="strictly increasing"):
        rul_hours_from_timestamps(ts)


def test_cap_flattens_early_life_only():
    capped = cap_rul(np.array([500.0, 200.0, 100.0, 10.0]), cap_hours=168.0)
    assert capped.tolist() == [168.0, 168.0, 100.0, 10.0]


def test_cap_rejects_nonpositive_horizon():
    with pytest.raises(ValueError):
        cap_rul(np.array([1.0]), cap_hours=0.0)


def test_build_rul_target_composes_both_steps():
    target = build_rul_target(_timestamps(400))  # 399h span, exceeds the cap
    assert target.max() == DEFAULT_RUL_CAP_HOURS
    assert target[-1] == 0.0


def test_aggregate_reduces_rows_to_snapshots():
    out = aggregate_windows_to_snapshot({"rms": np.arange(60, dtype=float)}, 20)
    assert out["rms"].shape == (3,)


def test_aggregate_takes_median_within_each_snapshot():
    feats = {"a": np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])}
    assert aggregate_windows_to_snapshot(feats, 3)["a"].tolist() == [2.0, 20.0]


def test_aggregate_median_resists_a_single_transient():
    """One spiking window must not move its snapshot's summary."""
    feats = {"a": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1e6])}
    assert aggregate_windows_to_snapshot(feats, 3)["a"].tolist() == [1.0, 1.0]


def test_aggregate_trims_incomplete_trailing_snapshot():
    out = aggregate_windows_to_snapshot({"a": np.arange(7, dtype=float)}, 3)
    assert out["a"].shape == (2,)


def test_aggregate_rejects_fewer_windows_than_one_snapshot():
    with pytest.raises(ValueError):
        aggregate_windows_to_snapshot({"a": np.arange(2, dtype=float)}, 5)