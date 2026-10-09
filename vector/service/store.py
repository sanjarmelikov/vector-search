"""The searchable state behind the API: documents, chunks, and the vector index.

Concurrency: many queries can search at once, but adding documents changes the
HNSW graph, so a write must wait for in-flight searches and block new ones
(a readers-writer lock). Python's GIL means two threads never run Python code
at the same instant, but a search can be paused mid-walk while another thread
runs, so without the lock a query could see a half-linked node.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from vector.chunking import Chunk, Chunker, chunk_documents
from vector.data.beir import Document
from vector.embed.base import Embedder
from vector.eval.metrics import dedupe_docs
from vector.index.hnsw import HNSWIndex
from vector.rerank import Reranker, rerank
from vector.service.batching import EmbeddingBatcher


class RWLock:
    """Many readers or one writer. Writers get priority, so a stream of queries can't starve an add."""

    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    @contextmanager
    def read(self):
        with self._cond:
            while self._writer or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self):
        with self._cond:
            self._writers_waiting += 1
            while self._writer or self._readers:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


@dataclass
class Hit:
    doc_id: str
    title: str
    score: float
    text: str  # the document's best-matching chunk


class SearchStore:
    def __init__(
        self,
        embedder: Embedder,
        chunker: Chunker,
        index: HNSWIndex | None = None,
        reranker: Reranker | None = None,
        k_chunks: int = 100,
        rerank_depth: int = 50,
        batch_queries: bool = True,
    ):
        self.embedder = embedder
        self.chunker = chunker
        self.index = index or HNSWIndex(embedder.dim)
        self.reranker = reranker
        self.k_chunks = k_chunks
        self.rerank_depth = rerank_depth
        self.chunks: list[Chunk] = []
        self.titles: dict[str, str] = {}
        self._lock = RWLock()
        # Model inference isn't guaranteed thread-safe (especially on Apple's GPU), so one at a time.
        self._model_lock = threading.Lock()
        # One add at a time, so two requests can't both add the same new document.
        self._add_lock = threading.Lock()
        # Concurrent queries get embedded together in one model call (see batching.py).
        self._batcher = (
            EmbeddingBatcher(embedder.embed, lock=self._model_lock) if batch_queries else None
        )

    # ---- writes ----

    def add_documents(self, docs: list[Document]) -> int:
        """Chunk, embed and index new documents (ids already present are skipped). Returns chunks added."""
        with self._add_lock:
            new = list({d.doc_id: d for d in docs if d.doc_id not in self.titles}.values())
            chunks = chunk_documents(new, self.chunker)
            if not chunks:
                return 0
            with self._model_lock:  # embedding happens outside the write lock: queries keep running
                vectors = self.embedder.embed([c.text for c in chunks])
            with self._lock.write():
                self.index.add(vectors)  # ids continue from len(index) = positions in self.chunks
                self.chunks.extend(chunks)
                for d in new:
                    self.titles[d.doc_id] = d.title
            return len(chunks)

    # ---- reads ----

    def query(self, text: str, k: int = 10, use_rerank: bool = False) -> tuple[list[Hit], dict[str, float]]:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        if self._batcher is not None:
            vector = self._batcher.embed(self.embedder.query_prefix + text)
        else:
            with self._model_lock:
                vector = self.embedder.embed([self.embedder.query_prefix + text])[0]
        timings["embed_ms"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        with self._lock.read():
            ids, scores = self.index.search(vector, self.k_chunks)
            hits = [(self.chunks[i], float(s)) for i, s in zip(ids.tolist(), scores.tolist())]
        timings["search_ms"] = (time.perf_counter() - start) * 1000

        if use_rerank and self.reranker is not None and hits:
            start = time.perf_counter()
            depth = min(self.rerank_depth, len(hits))
            with self._model_lock:
                rr_scores = self.reranker.score(text, [c.text for c, _ in hits[:depth]])
            order = rerank(list(range(len(hits))), rr_scores, depth)
            hits = [(hits[i][0], float(rr_scores[i]) if i < depth else hits[i][1]) for i in order]
            timings["rerank_ms"] = (time.perf_counter() - start) * 1000

        best: dict[str, tuple[Chunk, float]] = {}
        for chunk, score in hits:
            best.setdefault(chunk.doc_id, (chunk, score))  # first = best rank for that doc
        doc_ids = dedupe_docs(chunk.doc_id for chunk, _ in hits)[:k]
        results = [
            Hit(d, self.titles.get(d, ""), round(best[d][1], 4), best[d][0].text) for d in doc_ids
        ]
        return results, {name: round(ms, 2) for name, ms in timings.items()}

    def stats(self) -> dict:
        with self._lock.read():
            out = {"documents": len(self.titles), "chunks": len(self.chunks)}
        if self._batcher is not None and self._batcher.batches:
            out["mean_query_batch"] = round(self._batcher.texts / self._batcher.batches, 2)
        return out

    # ---- persistence ----

    def save(self, directory: Path) -> None:
        """Write the index and chunk table to `directory`, replacing any previous save atomically."""
        directory = Path(directory)
        tmp = directory.with_name(directory.name + ".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        with self._lock.read():
            self.index.save(tmp / "index.npz")
            meta = {
                "embedder": self.embedder.name,
                "chunker": self.chunker.name,
                "chunks": [asdict(c) for c in self.chunks],
                "titles": self.titles,
            }
        (tmp / "meta.json").write_text(json.dumps(meta))
        old = directory.with_name(directory.name + ".old")
        if directory.exists():
            directory.rename(old)
        tmp.rename(directory)  # readers see the old save or the new one, never a mix
        if old.exists():
            for f in old.iterdir():
                f.unlink()
            old.rmdir()

    @classmethod
    def load(cls, directory: Path, embedder: Embedder, chunker: Chunker, **kwargs) -> SearchStore:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        if meta["embedder"] != embedder.name or meta["chunker"] != chunker.name:
            raise ValueError(
                f"saved index was built with {meta['embedder']} / {meta['chunker']}, "
                f"not {embedder.name} / {chunker.name}; rebuild it"
            )
        store = cls(embedder, chunker, index=HNSWIndex.load(directory / "index.npz"), **kwargs)
        store.chunks = [Chunk(**c) for c in meta["chunks"]]
        store.titles = meta["titles"]
        if len(store.chunks) != len(store.index):
            raise ValueError("saved chunk table and index disagree; rebuild it")
        return store


def documents_from_payload(items: list[dict]) -> list[Document]:
    return [Document(str(i["id"]), i.get("title", ""), i["text"]) for i in items]


def as_dicts(hits: list[Hit]) -> list[dict]:
    return [asdict(h) for h in hits]

