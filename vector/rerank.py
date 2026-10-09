"""Second-stage re-ranking with a cross-encoder.

Embedding search scores a query and a passage that were turned into vectors
*separately*. A cross-encoder reads them *together*, as one input, and outputs
one relevance score, so it can check whether the passage actually answers
the query. That's far more accurate, but it needs a full model run per
(query, passage) pair, so it only re-orders the top few results of the fast search.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Relevance of each passage to the query (higher = more relevant), in input order."""
        ...


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32):
        # Imported here so the rest of the package works without torch installed.
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model)
        self.name = model
        self.batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.empty(0, dtype=np.float32)
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)


def rerank(ids: list[int], scores: np.ndarray, depth: int) -> list[int]:
    """Re-order the first `depth` ids by score (best first); anything below depth keeps its place."""
    head = [ids[i] for i in np.argsort(-scores[:depth], kind="stable")]
    return head + list(ids[depth:])


RERANKERS = {
    "ms-marco-minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # 22M params, trained on web search
    "bge-reranker": "BAAI/bge-reranker-base",  # 278M params, trained for retrieval broadly
}
