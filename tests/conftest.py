import numpy as np
import pytest


class BagOfWordsEmbedder:
    """Deterministic stand-in for a real model: one dimension per vocabulary word.

    Retrieval results are easy to work out by hand, and it counts calls so
    tests can tell cache hits from misses.
    """

    def __init__(self, vocab: list[str], name: str = "bow"):
        self.vocab = {word: i for i, word in enumerate(vocab)}
        self.name = name
        self.dim = len(vocab)
        self.max_tokens = 3
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.split():
                if word in self.vocab:
                    vectors[row, self.vocab[word]] += 1
        return vectors

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(t.split()) for t in texts]


@pytest.fixture
def bow():
    return BagOfWordsEmbedder(["apple", "banana", "cherry", "date"])
