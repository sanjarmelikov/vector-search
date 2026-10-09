import numpy as np
import pytest

from vector.data.sift import read_fvecs


def write_fvecs(path, vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    dims = np.full((len(vectors), 1), vectors.shape[1], dtype=np.int32)
    np.hstack([dims, vectors.view(np.int32)]).tofile(path)


def test_read_fvecs_round_trip_and_prefix(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    write_fvecs(tmp_path / "x.fvecs", data)
    assert np.array_equal(read_fvecs(tmp_path / "x.fvecs"), data)
    assert np.array_equal(read_fvecs(tmp_path / "x.fvecs", n=2), data[:2])
    assert read_fvecs(tmp_path / "x.fvecs").dtype == np.float32


def test_read_fvecs_rejects_garbage(tmp_path):
    # Says dimension 4, but the second row's header is 9.
    np.array([4, 0, 0, 0, 0, 9, 0, 0, 0, 0], dtype=np.int32).tofile(tmp_path / "bad.fvecs")
    with pytest.raises(ValueError):
        read_fvecs(tmp_path / "bad.fvecs")
