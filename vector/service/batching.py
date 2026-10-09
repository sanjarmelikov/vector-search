"""Micro-batching for query embeddings.

Embedding one query takes ~13 ms on this laptop, and models run one call at a
time, so a server embedding queries one by one tops out near 75-90 queries/s
no matter how many clients wait. But a batch of 16 queries costs barely more
than one (the GPU does them in parallel). So: requests drop their text in a
queue; one background thread takes whatever is waiting (up to `max_batch`,
waiting at most `max_wait_ms` for more to arrive), embeds them in one call,
and hands each request its own vector.

Under light load a request waits at most `max_wait_ms` extra; under heavy load
batches fill up and throughput rises with them.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future

import numpy as np


class EmbeddingBatcher:
    def __init__(self, embed, max_batch: int = 32, max_wait_ms: float = 2.0, lock: threading.Lock | None = None):
        self._embed = embed  # function: list[str] -> (n, dim) array
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self._lock = lock or threading.Lock()  # shared with any other user of the same model
        self._queue: queue.Queue[tuple[str, Future]] = queue.Queue()
        self.batches = 0  # stats: how many model calls, and how many texts they covered
        self.texts = 0
        self._thread = threading.Thread(target=self._loop, name="embedding-batcher", daemon=True)
        self._thread.start()

    def embed(self, text: str, timeout: float = 30.0) -> np.ndarray:
        """Embed one text (blocking); it gets batched with whatever else is waiting."""
        future: Future = Future()
        self._queue.put((text, future))
        return future.result(timeout=timeout)

    def _loop(self) -> None:
        while True:
            batch = [self._queue.get()]  # sleep until there's at least one request
            deadline = time.perf_counter() + self.max_wait
            while len(batch) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            texts = [t for t, _ in batch]
            try:
                with self._lock:
                    vectors = self._embed(texts)
            except Exception as exc:  # noqa: BLE001 - every waiting request gets the error
                for _, f in batch:
                    f.set_exception(exc)
                continue
            self.batches += 1
            self.texts += len(batch)
            for (_, f), v in zip(batch, vectors):
                f.set_result(v)
