import numpy as np
import pandas as pd
import pytest

from bearing_pdm.feature_matrix import build_sensor_matrix


def _frame():
    """Two bearings, rows deliberately shuffled. The feature equals the time
    step, so after correct sorting it must increase within each bearing."""
    rows = []
    for unit, scale in [("b1", 1.0), ("b2", 100.0)]:
        for t in range(10):
            rows.append({"bearing_id": unit, "snapshot_ts": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=t),
                         "f1": t * scale, "f2": (t % 3) * scale})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_rows_come_back_sorted_and_aligned_with_the_matrix():
    ordered, matrix = build_sensor_matrix(_frame(), ["f1", "f2"], healthy_fraction=0.3)
    assert matrix.shape == (20, 2)
    assert list(ordered["bearing_id"]) == ["b1"] * 10 + ["b2"] * 10
    for unit in ["b1", "b2"]:
        rows = (ordered["bearing_id"] == unit).to_numpy()
        assert np.all(np.diff(matrix[rows, 0]) > 0)


def test_each_bearing_is_normalized_independently():
    """b2 is 100x b1 in raw units; after per-bearing normalization they match."""
    ordered, matrix = build_sensor_matrix(_frame(), ["f1", "f2"], healthy_fraction=0.3)
    b1 = matrix[(ordered["bearing_id"] == "b1").to_numpy()]
    b2 = matrix[(ordered["bearing_id"] == "b2").to_numpy()]
    assert np.allclose(b1, b2)


def test_missing_column_is_reported():
    with pytest.raises(KeyError, match="f3"):
        build_sensor_matrix(_frame(), ["f1", "f3"])