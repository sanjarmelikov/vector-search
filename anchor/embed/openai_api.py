"""OpenAI embedding API, batched by input count and token budget."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
# Per-request limits from the API docs: 2048 inputs and 300k tokens in total.
# The token budget leaves headroom because our count is an estimate.
MAX_BATCH_INPUTS = 2048
MAX_BATCH_TOKENS = 250_000


def _tiktoken_counter(texts: list[str]) -> list[int]:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")  # the text-embedding-3 tokenizer
    return [len(ids) for ids in enc.encode_batch(texts)]


class OpenAIEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client=None,
        token_counter: Callable[[list[str]], list[int]] = _tiktoken_counter,
        max_retries: int = 6,
    ):
        if model not in DIMS:
            raise ValueError(f"unknown model {model!r}; expected one of {sorted(DIMS)}")
        if client is None:
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv()  # reads OPENAI_API_KEY from .env
            # The SDK retries rate limits, timeouts and 5xx errors with exponential backoff.
            client = OpenAI(max_retries=max_retries)
        self._client = client
        self._count = token_counter
        self.name = model
        self.dim = DIMS[model]
        self.max_tokens = 8191

    def count_tokens(self, texts: list[str]) -> list[int]:
        return self._count(texts)

    def _batches(self, texts: list[str]) -> Iterator[list[str]]:
        batch: list[str] = []
        batch_tokens = 0
        for text, n in zip(texts, self.count_tokens(texts)):
            if n > self.max_tokens:
                # The API rejects these instead of truncating; chunk the text first.
                raise ValueError(f"text has {n} tokens, over the {self.max_tokens} limit")
            if batch and (len(batch) == MAX_BATCH_INPUTS or batch_tokens + n > MAX_BATCH_TOKENS):
                yield batch
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += n
        if batch:
            yield batch

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for batch in self._batches(texts):
            response = self._client.embeddings.create(model=self.name, input=batch)
            # Each item carries its position in the batch; don't trust response order.
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return np.asarray(vectors, dtype=np.float32).reshape(len(texts), self.dim)
