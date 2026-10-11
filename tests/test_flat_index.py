import numpy as np
import pytest

from vector.index import FlatIndex


def test_matches_brute_force_cosine():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 32)).astype(np.float32)
    query = rng.normal(size=32).astype(np.float32)
    index = FlatIndex(32)
    index.add(data[:200])
    index.add(data[200:])  # ids continue across add() calls

    ids, scores = index.search(query, k=10)

    cos = data @ query / (np.linalg.norm(data, axis=1) * np.linalg.norm(query))
    expected = np.argsort(-cos)[:10]
    assert ids.tolist() == expected.tolist()
    assert np.allclose(scores, cos[expected], atol=1e-5)
    assert len(index) == 500


def test_k_larger_than_index_and_empty_index():
    index = FlatIndex(4)
    assert len(index.search(np.ones(4), k=5)[0]) == 0
    index.add(np.eye(4))
    ids, _ = index.search(np.array([1, 0, 0, 0]), k=10)
    assert len(ids) == 4 and ids[0] == 0


def test_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        FlatIndex(4).add(np.ones((2, 3)))
