"""Local embedding models via sentence-transformers (runs on CPU or Apple MPS)."""

from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str | None = None,
        query_prefix: str = "",
    ):
        # Imported here so the rest of the package works without torch installed.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model, device=device)
        self.batch_size = batch_size
        self.name = model
        self.query_prefix = query_prefix
        self.dim = self._model.get_embedding_dimension()
        # The model silently drops every token past this limit.
        self.max_tokens = self._model.max_seq_length

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), self.dim)

    def count_tokens(self, texts: list[str]) -> list[int]:
        # Includes the [CLS]/[SEP] special tokens, which also count toward the limit.
        encoded = self._model.tokenizer(texts, truncation=False, verbose=False)
        return [len(ids) for ids in encoded["input_ids"]]
