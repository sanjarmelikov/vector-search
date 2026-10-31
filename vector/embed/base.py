"""Common interface for every embedding model (local or API)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    # Identifies the model in cache keys and results; must change whenever
    # the vectors would change.
    name: str
    dim: int
    # Inputs longer than this many tokens are truncated (or rejected).
    max_tokens: int
    # Prepended to queries (not documents) before embedding. Some retrieval
    # models are trained with an instruction like this; "" for most.
    query_prefix: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim), in input order."""
        ...

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Token count of each text under this model's tokenizer."""
        ...
