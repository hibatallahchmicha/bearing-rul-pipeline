import numpy as np

from bearing_pdm.ingestion import list_snapshot_files, read_snapshot


def test_files_sorted_chronologically(tmp_path):
    names = ["2003.10.22.12.26.24", "2003.10.22.12.06.24", "2003.10.22.12.16.24"]
    for name in names:
        np.savetxt(tmp_path / name, np.random.randn(10, 4))

    files = list_snapshot_files(tmp_path)
    assert [f.name for f in files] == sorted(names)


def test_read_snapshot_shape(tmp_path):
    path = tmp_path / "2003.10.22.12.06.24"
    np.savetxt(path, np.random.randn(50, 4))
    assert read_snapshot(path).shape == (50, 4)
