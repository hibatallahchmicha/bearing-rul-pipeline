import numpy as np
import pytest

from bearing_pdm.segmentation import window_signal


def test_window_shape_and_overlap():
    windows = window_signal(np.arange(20480))
    assert windows.shape == (19, 2048)
    assert windows[1][0] == 1024  # 50% overlap -> hop of 1024


def test_rejects_signal_shorter_than_window():
    with pytest.raises(ValueError):
        window_signal(np.arange(100))


def test_rejects_2d_input():
    with pytest.raises(ValueError):
        window_signal(np.zeros((10, 10)))
