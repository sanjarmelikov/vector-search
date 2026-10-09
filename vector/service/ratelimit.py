"""Rate limiting: a token bucket per client, behind an interface Throttle can replace.

Each client has a bucket holding up to `capacity` tokens that refills at `rate`
tokens per second. A request costs some tokens; if the bucket has enough, they're
spent and the request goes through, otherwise it's rejected with a "retry after"
time. Bursts up to `capacity` are allowed; the long-run average can't exceed `rate`.

This in-process limiter only protects one server process. A shared limiter
(the planned Throttle service, backed by Redis) would enforce one budget
across many servers; `RateLimiter` is the seam where it plugs in.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Decision:
    allowed: bool
    remaining: float  # tokens left after this request
    retry_after: float  # seconds until enough tokens for this request (0 if allowed)


class RateLimiter(Protocol):
    def allow(self, client: str, cost: int = 1) -> Decision: ...


class TokenBucketLimiter:
    def __init__(self, rate: float, capacity: float, clock: Callable[[], float] = time.monotonic):
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = rate
        self.capacity = capacity
        self._clock = clock  # injectable, so tests control time instead of sleeping
        self._buckets: dict[str, tuple[float, float]] = {}  # client -> (tokens, last refill time)
        self._lock = threading.Lock()

    def allow(self, client: str, cost: int = 1) -> Decision:
        if cost > self.capacity:
            return Decision(False, 0.0, float("inf"))  # can never be afforded
        with self._lock:  # read-refill-spend must be atomic, or two threads could spend the same tokens
            now = self._clock()
            tokens, last = self._buckets.get(client, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)  # lazy refill
            if tokens >= cost:
                self._buckets[client] = (tokens - cost, now)
                return Decision(True, tokens - cost, 0.0)
            self._buckets[client] = (tokens, now)
            return Decision(False, tokens, (cost - tokens) / self.rate)
