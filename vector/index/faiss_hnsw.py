"""Meta's FAISS HNSW behind our index interface: the C++ yardstick for our Python HNSW.

Same algorithm and the same knobs (M, ef_construction, ef_search), so any speed
gap between the two is implementation (C++ vs Python), not algorithm.
"""

from __future__ import annotations

import numpy as np

from vector.index.base import normalize


class FaissHNSWIndex:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        import faiss  # optional dependency: pip install -e '.[bench]'

        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        # Inner product on normalized vectors = cosine, matching FlatIndex and HNSWIndex.
        self._index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = ef_construction

    def add(self, vectors: np.ndarray) -> None:
        vectors = normalize(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {vectors.shape}")
        self._index.add(np.ascontiguousarray(vectors))

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self))
        if k <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        self._index.hnsw.efSearch = max(self.ef_search, k)  # same rule as HNSWIndex
        scores, ids = self._index.search(normalize(query).reshape(1, -1), k)
        keep = ids[0] >= 0  # FAISS pads with -1 when it finds fewer than k
        return ids[0][keep].astype(np.int64), scores[0][keep]

    def __len__(self) -> int:
        return self._index.ntotal
