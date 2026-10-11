import random
from types import SimpleNamespace

import numpy as np
import pytest

from vector.embed import openai_api
from vector.embed.openai_api import OpenAIEmbedder


class FakeClient:
    """Mimics client.embeddings.create; returns items shuffled to test reordering."""

    def __init__(self, dim: int):
        self.dim = dim
        self.batches: list[list[str]] = []
        self.embeddings = self

    def create(self, model: str, input: list[str]):
        self.batches.append(list(input))
        # Encode each text's number in the first coordinate so order is checkable.
        data = [
            SimpleNamespace(index=i, embedding=[float(text.split()[-1])] + [0.0] * (self.dim - 1))
            for i, text in enumerate(input)
        ]
        random.Random(0).shuffle(data)
        return SimpleNamespace(data=data)


def word_counter(texts):
    return [len(t.split()) for t in texts]


@pytest.fixture
def embedder():
    return OpenAIEmbedder(client=FakeClient(1536), token_counter=word_counter)


def test_preserves_input_order_across_batches(embedder, monkeypatch):
    monkeypatch.setattr(openai_api, "MAX_BATCH_INPUTS", 3)
    texts = [f"text {i}" for i in range(8)]
    vectors = embedder.embed(texts)
    assert vectors.shape == (8, 1536) and vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors[:, 0], np.arange(8))
    assert [len(b) for b in embedder._client.batches] == [3, 3, 2]


def test_batches_respect_token_budget(embedder, monkeypatch):
    monkeypatch.setattr(openai_api, "MAX_BATCH_TOKENS", 5)
    embedder.embed(["a b 1", "c 2", "d e f 3", "4"])  # 3, 2, 4, 1 tokens
    assert [len(b) for b in embedder._client.batches] == [2, 2]


def test_rejects_text_over_token_limit(embedder):
    embedder.max_tokens = 3
    with pytest.raises(ValueError, match="over the 3 limit"):
        embedder.embed(["a b c d"])


def test_unknown_model():
    with pytest.raises(ValueError, match="unknown model"):
        OpenAIEmbedder("text-embedding-9", client=FakeClient(1))
