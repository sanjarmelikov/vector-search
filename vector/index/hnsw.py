"""HNSW: approximate nearest-neighbor search over a layered graph, from scratch.

Follows Malkov & Yashunin (2018), "Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs". Scores are
cosine similarities (higher = closer), exactly as in FlatIndex, so the two are
interchangeable and FlatIndex serves as the answer key.

Lesson map (notes/STUDY_GUIDE.md, Phase 3):
    _search_layer      Lessons 1–2: greedy walk (ef=1) and beam search (ef>1)
    _insert            Lesson 3: build the graph by searching for each new point's neighbors
    _select_neighbors  Lesson 4: keep neighbors that point in different directions
    _random_level      Lesson 5: layers, the "highways"
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from vector.index.base import normalize


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 0,
    ):
        if M < 2:
            raise ValueError("M must be at least 2")
        self.dim = dim
        self.M = M  # links per node on layers >= 1
        self.M0 = 2 * M  # layer 0 holds every node, so it allows more links
        self.ef_construction = ef_construction  # beam width while building
        self.ef_search = ef_search  # beam width while querying
        self._level_mult = 1 / math.log(M)  # "mL" in the paper
        self._rng = np.random.default_rng(seed)

        self._vectors = np.empty((16, dim), dtype=np.float32)  # capacity doubles as needed
        self._n = 0
        self._levels: list[int] = []  # top layer of each node
        # _layers[l][node] = neighbor ids of `node` on layer l. Layer 0 contains every node.
        self._layers: list[dict[int, list[int]]] = []
        self._entry: int | None = None  # a node on the top layer; every search starts here

        self._scored = 0  # running count of similarity computations
        self.scored_last_search = 0  # how many vectors the latest search() compared against

    # ---- public interface (same as FlatIndex) ----

    def add(self, vectors: np.ndarray) -> None:
        vectors = normalize(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected shape (n, {self.dim}), got {vectors.shape}")
        start, end = self._n, self._n + len(vectors)
        if end > len(self._vectors):
            grown = np.empty((max(end, 2 * len(self._vectors)), self.dim), dtype=np.float32)
            grown[:start] = self._vectors[:start]
            self._vectors = grown
        self._vectors[start:end] = vectors
        for node in range(start, end):
            self._n = node + 1
            self._insert(node)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._entry is None or k <= 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        query = normalize(query)
        self._scored = 0
        # Highways first: on each upper layer, a greedy walk (ef=1) gets closer to the query.
        entry = [self._entry]
        for layer in range(self._levels[self._entry], 0, -1):
            entry = [self._search_layer(query, entry, 1, layer)[0][1]]
        # Then a wide beam search on layer 0, which holds every node.
        found = self._search_layer(query, entry, max(self.ef_search, k), 0)[:k]
        self.scored_last_search = self._scored
        ids = np.array([node for _, node in found], dtype=np.int64)
        scores = np.array([score for score, _ in found], dtype=np.float32)
        return ids, scores

    def __len__(self) -> int:
        return self._n

    # ---- the algorithm ----

    def _search_layer(
        self, query: np.ndarray, entry: list[int], ef: int, layer: int
    ) -> list[tuple[float, int]]:
        """Beam search on one layer. Returns the best `ef` nodes found, as (score, id), best first.

        With ef=1 this is the greedy walk from Lesson 1: move to the best neighbor
        until none is better. A larger ef keeps several candidates alive, so one
        dead end (local maximum) doesn't end the search.
        """
        graph = self._layers[layer]
        entry_scores = (self._vectors[entry] @ query).tolist()
        self._scored += len(entry)
        visited = set(entry)
        # candidates: nodes whose neighbors we haven't explored yet, best first.
        # heapq is a min-heap, so scores are negated to pop the best one.
        candidates = [(-s, node) for s, node in zip(entry_scores, entry)]
        heapq.heapify(candidates)
        # results: the best ef nodes seen so far. A min-heap, so results[0] is the worst of them.
        results = [(s, node) for s, node in zip(entry_scores, entry)]
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)

        while candidates:
            neg_score, node = heapq.heappop(candidates)
            if -neg_score < results[0][0]:
                break  # the best unexplored node is worse than everything we're keeping: done
            fresh = [n for n in graph[node] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            scores = (self._vectors[fresh] @ query).tolist()  # one matrix-vector product
            self._scored += len(fresh)
            for s, n in zip(scores, fresh):
                if len(results) < ef or s > results[0][0]:
                    heapq.heappush(candidates, (-s, n))
                    heapq.heappush(results, (s, n))
                    if len(results) > ef:
                        heapq.heappop(results)  # drop the worst
        return sorted(results, reverse=True)

    def _select_neighbors(
        self, base: np.ndarray, candidates: list[tuple[float, int]], m: int
    ) -> list[int]:
        """Choose up to m neighbors from candidates (best first) that cover different directions.

        A candidate is kept only if it is closer to `base` than to every neighbor
        already kept. If it's closer to one of them, that neighbor already "covers"
        it: the walk can reach it in one more hop. This keeps long-range links that
        plain "take the m closest" would crowd out (the paper's Algorithm 4).
        """
        selected: list[int] = []
        for score, c in candidates:
            if len(selected) == m:
                break
            if not selected or score > float(np.max(self._vectors[selected] @ self._vectors[c])):
                selected.append(c)
        return selected

    def _random_level(self) -> int:
        """Top layer for a new node: 0 for most nodes, each higher layer about M times rarer."""
        return int(-math.log(1.0 - self._rng.random()) * self._level_mult)

    def _insert(self, node: int) -> None:
        query = self._vectors[node]
        level = self._random_level()
        self._levels.append(level)
        while len(self._layers) <= level:
            self._layers.append({})
        for layer in range(level + 1):
            self._layers[layer][node] = []

        if self._entry is None:  # the very first node
            self._entry = node
            return

        top = self._levels[self._entry]
        entry = [self._entry]
        # 1. Above the new node's own level: greedy descent, only to find a good starting point.
        for layer in range(top, level, -1):
            entry = [self._search_layer(query, entry, 1, layer)[0][1]]
        # 2. On every layer the node lives on: find close nodes, link to a few, and link back.
        for layer in range(min(level, top), -1, -1):
            found = self._search_layer(query, entry, self.ef_construction, layer)
            neighbors = self._select_neighbors(query, found, self.M)
            self._layers[layer][node] = neighbors
            max_links = self.M0 if layer == 0 else self.M
            for n in neighbors:
                links = self._layers[layer][n]
                links.append(node)
                if len(links) > max_links:  # too many links: re-choose n's neighbors
                    scores = (self._vectors[links] @ self._vectors[n]).tolist()
                    ranked = sorted(zip(scores, links), reverse=True)
                    self._layers[layer][n] = self._select_neighbors(self._vectors[n], ranked, max_links)
            entry = [n for _, n in found]  # this layer's results seed the layer below

        if level > top:
            self._entry = node  # the new node reaches higher than anyone: it's the new entry
