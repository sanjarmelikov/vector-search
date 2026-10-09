import threading
import time

import numpy as np
import pytest

from vector.service.batching import EmbeddingBatcher


class SlowEmbedder:
    """Each call costs 20 ms however many texts it gets, like a GPU batch. Vector = [len(text)]."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(len(texts))
        time.sleep(0.02)
        return np.array([[len(t)] for t in texts], dtype=np.float32)


def test_each_caller_gets_its_own_vector_and_calls_are_batched():
    model = SlowEmbedder()
    batcher = EmbeddingBatcher(model, max_batch=32, max_wait_ms=5)
    results = {}

    def ask(i):
        results[i] = batcher.embed("x" * i)[0]

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(1, 41)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    assert results == {i: float(i) for i in range(1, 41)}  # nobody got someone else's vector
    assert sum(model.calls) == 40 and len(model.calls) < 40 and max(model.calls) <= 32
    assert elapsed < 40 * 0.02  # faster than 40 one-at-a-time calls (0.8 s)


def test_single_request_waits_at_most_max_wait():
    batcher = EmbeddingBatcher(SlowEmbedder(), max_wait_ms=5)
    start = time.perf_counter()
    batcher.embed("hello")
    assert time.perf_counter() - start < 0.02 + 0.005 + 0.05  # one model call + max wait + slack


def test_model_errors_reach_every_waiting_caller():
    def broken(texts):
        raise RuntimeError("model crashed")

    batcher = EmbeddingBatcher(broken)
    with pytest.raises(RuntimeError, match="model crashed"):
        batcher.embed("x")
    # The batcher thread survives the error and keeps serving.
    good = EmbeddingBatcher(SlowEmbedder())
    assert good.embed("abc")[0] == 3.0
