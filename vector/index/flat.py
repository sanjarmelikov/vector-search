"""Exact brute-force search: the correctness and latency baseline."""

from __future__ import annotations

import numpy as np

from vector.index.base import normalize


class FlatIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = np.empty((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        vectors = normalize(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {vectors.shape}")
        self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self._vectors @ normalize(query)
        k = min(k, len(scores))
        if k == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        # argpartition is O(n); only the k winners get fully sorted.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return top.astype(np.int64), scores[top]

    def __len__(self) -> int:
        return len(self._vectors)
