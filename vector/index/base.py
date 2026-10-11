"""Common interface for every vector index (exact, from-scratch HNSW, FAISS)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product equals cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class VectorIndex(Protocol):
    dim: int

    def add(self, vectors: np.ndarray) -> None:
        """Append vectors; their ids are their insertion positions."""
        ...

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids, scores) of the top-k by cosine similarity, best first."""
        ...

    def __len__(self) -> int: ...
