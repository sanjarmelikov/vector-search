import numpy as np

from vector.eval.bench import benchmark, exact_top_k
from vector.index import FlatIndex


def test_exact_top_k_matches_flat_index_across_batches():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(500, 16)).astype(np.float32)
    queries = rng.normal(size=(7, 16)).astype(np.float32)
    flat = FlatIndex(16)
    flat.add(base)
    truth = exact_top_k(base, queries, k=5, batch=3)  # batch smaller than #queries
    for q, ids in zip(queries, truth):
        assert ids.tolist() == flat.search(q, 5)[0].tolist()


def test_benchmark_exact_index_has_perfect_recall():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(300, 8)).astype(np.float32)
    queries = rng.normal(size=(20, 8)).astype(np.float32)
    flat = FlatIndex(8)
    flat.add(base)
    row = benchmark(flat, queries, exact_top_k(base, queries, k=10), k=10)
    assert row["recall@10"] == 1.0 and row["qps"] > 0 and row["scored_mean"] is None
