import numpy as np
import pytest

from vector.index import FlatIndex

faiss_hnsw = pytest.importorskip("vector.index.faiss_hnsw", exc_type=ImportError)


def test_faiss_hnsw_matches_exact_search():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2000, 32)).astype(np.float32)
    exact = FlatIndex(32)
    exact.add(data)
    index = faiss_hnsw.FaissHNSWIndex(32, M=16, ef_construction=100, ef_search=64)
    index.add(data[:1000])
    index.add(data[1000:])
    assert len(index) == 2000
    hits = 0
    for q in rng.normal(size=(50, 32)).astype(np.float32):
        ids, scores = index.search(q, 10)
        true_ids, true_scores = exact.search(q, 10)
        hits += len(set(ids.tolist()) & set(true_ids.tolist()))
        assert np.all(np.diff(scores) <= 1e-6)  # best first, same orientation as ours
    assert hits / 500 >= 0.95


def test_faiss_hnsw_small_and_empty():
    index = faiss_hnsw.FaissHNSWIndex(4, M=4)
    assert len(index.search(np.ones(4), k=5)[0]) == 0
    index.add(np.eye(4))
    ids, _ = index.search(np.array([1, 0, 0, 0]), k=10)
    assert ids[0] == 0 and len(ids) == 4
