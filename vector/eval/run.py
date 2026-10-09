"""Run one retrieval experiment end to end and record the results.

    python -m vector.eval.run --model minilm --chunker whole fixed-200-40
    python -m vector.eval.run --model bge-small --chunker whole --index flat hnsw-16-200-64

Each (model, chunker, index) combination becomes one JSON row in the --out file
(default results/phase1.jsonl), tagged with the git commit that produced it.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vector.chunking import (
    Chunk,
    Chunker,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    WholeDocumentChunker,
    chunk_documents,
)
from vector.data.beir import BeirDataset, load_scifact
from vector.embed.base import Embedder
from vector.embed.cache import DEFAULT_CACHE_DIR, cached_embed
from vector.eval.metrics import dedupe_docs, evaluate, per_query_ndcg
from vector.index import FlatIndex, HNSWIndex, VectorIndex
from vector.rerank import RERANKERS, Reranker, rerank

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results" / "phase1.jsonl"
ANN_K = 10


def _minilm() -> Embedder:
    from vector.embed.local import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")


def _bge_small() -> Embedder:
    from vector.embed.local import SentenceTransformerEmbedder

    # BGE was trained to see this instruction in front of short retrieval queries.
    return SentenceTransformerEmbedder(
        "BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
    )


def _openai_small() -> Embedder:
    from vector.embed.openai_api import OpenAIEmbedder

    return OpenAIEmbedder("text-embedding-3-small")


def _load_reranker(name: str) -> Reranker:
    from vector.rerank import CrossEncoderReranker

    return CrossEncoderReranker(RERANKERS[name])


MODELS: dict[str, Callable[[], Embedder]] = {
    "minilm": _minilm,
    "bge-small": _bge_small,
    "openai-small": _openai_small,
}

_CHUNKERS = {"fixed": FixedSizeChunker, "sentence": SentenceChunker, "recursive": RecursiveChunker}


def parse_chunker(spec: str) -> Chunker:
    """'whole' or '<fixed|sentence|recursive>-<size>-<overlap>', e.g. 'fixed-200-40'."""
    if spec == "whole":
        return WholeDocumentChunker()
    kind, _, rest = spec.partition("-")
    size, _, overlap = rest.partition("-")
    if kind not in _CHUNKERS or not size.isdigit() or not overlap.isdigit():
        raise ValueError(f"bad chunker spec {spec!r}; try 'whole' or 'fixed-200-40'")
    return _CHUNKERS[kind](int(size), int(overlap))


def parse_index(spec: str, dim: int) -> VectorIndex:
    """'flat', or '<hnsw|faiss-hnsw>-<M>-<ef_construction>-<ef_search>', e.g. 'hnsw-16-200-64'."""
    if spec == "flat":
        return FlatIndex(dim)
    parts = spec.split("-")
    kind, params = "-".join(parts[:-3]), parts[-3:]
    if kind not in ("hnsw", "faiss-hnsw") or len(params) != 3 or not all(p.isdigit() for p in params):
        raise ValueError(f"bad index spec {spec!r}; try 'flat', 'hnsw-16-200-64' or 'faiss-hnsw-16-200-64'")
    m, ef_construction, ef_search = map(int, params)
    if kind == "faiss-hnsw":
        from vector.index.faiss_hnsw import FaissHNSWIndex  # optional dependency

        return FaissHNSWIndex(dim, M=m, ef_construction=ef_construction, ef_search=ef_search)
    return HNSWIndex(dim, M=m, ef_construction=ef_construction, ef_search=ef_search)


def graph_key(spec: str) -> str:
    """Specs that differ only in ef_search (a query-time setting) share one built graph."""
    return spec if spec == "flat" else spec.rsplit("-", 1)[0]


def build_index(spec: str, vectors: np.ndarray, built: dict | None = None) -> tuple[VectorIndex, float]:
    """Build `spec` over vectors, or reuse a graph from `built` that differs only in ef_search."""
    index = parse_index(spec, vectors.shape[1])
    key = graph_key(spec)
    if built is not None and key in built:
        cached, build_seconds = built[key]
        if hasattr(index, "ef_search"):
            cached.ef_search = index.ef_search
        return cached, build_seconds
    start = time.perf_counter()
    index.add(vectors)
    build_seconds = time.perf_counter() - start
    if built is not None:
        built[key] = (index, build_seconds)
    return index, build_seconds


@dataclass
class Prepared:
    """Everything an index needs for one (model, chunker) pair, embedded once."""

    chunks: list[Chunk]
    chunk_vectors: np.ndarray
    query_ids: list[str]
    query_vectors: np.ndarray
    tokens: np.ndarray


def prepare(
    dataset: BeirDataset, chunker: Chunker, embedder: Embedder, cache_dir: Path = DEFAULT_CACHE_DIR
) -> Prepared:
    chunks = chunk_documents(list(dataset.corpus.values()), chunker)
    texts = [c.text for c in chunks]
    tokens = np.array(embedder.count_tokens(texts))
    chunk_vectors = cached_embed(embedder, texts, cache_dir)
    query_ids = list(dataset.queries)
    query_texts = [embedder.query_prefix + dataset.queries[q] for q in query_ids]
    query_vectors = cached_embed(embedder, query_texts, cache_dir)
    return Prepared(chunks, chunk_vectors, query_ids, query_vectors, tokens)


def evaluate_index(
    dataset: BeirDataset,
    chunker: Chunker,
    embedder: Embedder,
    prepared: Prepared,
    index_spec: str = "flat",
    k_chunks: int = 100,
    built: dict | None = None,
    per_query: bool = False,
    reranker: Reranker | None = None,
    rerank_depth: int = 50,
) -> dict:
    """Score one index over prepared vectors.

    `built` caches indexes across calls: specs that differ only in ef_search
    (a query-time setting) reuse the same graph instead of rebuilding it.
    """
    chunks, query_vectors = prepared.chunks, prepared.query_vectors

    index, build_seconds = build_index(index_spec, prepared.chunk_vectors, built)

    # Exact search is the answer key for approximate indexes (ANN recall).
    exact = FlatIndex(embedder.dim)
    exact.add(prepared.chunk_vectors)

    # One untimed search first, so one-off setup costs don't land in the latency numbers.
    index.search(query_vectors[0], k_chunks)

    results: dict[str, list[str]] = {}
    latencies, rerank_latencies, scored = [], [], []
    found_true = total_true = 0
    for query_id, vector in zip(prepared.query_ids, query_vectors):
        start = time.perf_counter()
        ids, _ = index.search(vector, k_chunks)
        latencies.append(time.perf_counter() - start)
        scored.append(getattr(index, "scored_last_search", len(index)))
        true_top = set(exact.search(vector, ANN_K)[0].tolist())
        found_true += len(true_top & set(ids[:ANN_K].tolist()))
        total_true += len(true_top)
        ids = ids.tolist()
        if reranker is not None:
            # The cross-encoder reads the raw query (no embedding prefix) with each chunk's text.
            start = time.perf_counter()
            scores = reranker.score(dataset.queries[query_id], [chunks[i].text for i in ids[:rerank_depth]])
            ids = rerank(ids, scores, rerank_depth)
            rerank_latencies.append(time.perf_counter() - start)
        # Several chunks can share a document; keep each doc's best rank.
        results[query_id] = dedupe_docs(chunks[i].doc_id for i in ids)

    tokens = prepared.tokens
    latencies_ms = np.array(latencies) * 1000
    row = {
        "model": embedder.name,
        "chunker": chunker.name,
        "index": index_spec,
        "n_docs": len(dataset.corpus),
        "n_chunks": len(chunks),
        "n_queries": len(prepared.query_ids),
        "dim": embedder.dim,
        "k_chunks": k_chunks,
        "tokens_mean": round(float(tokens.mean()), 1),
        "tokens_max": int(tokens.max()),
        "max_tokens": embedder.max_tokens,
        "query_prefix": embedder.query_prefix,
        # Truncation: how much of the corpus the model never actually reads.
        "truncated_frac": round(float(np.mean(tokens > embedder.max_tokens)), 4),
        "metrics": {name: round(value, 4) for name, value in evaluate(results, dataset.qrels).items()},
        # Share of the exact top-10 chunks the index also returned in its top 10.
        "ann_recall@10": round(found_true / total_true, 4),
        "scored_mean": round(float(np.mean(scored)), 1),
        "search_p50_ms": round(float(np.percentile(latencies_ms, 50)), 4),
        "search_p99_ms": round(float(np.percentile(latencies_ms, 99)), 4),
        "build_seconds": round(build_seconds, 4),
        "rerank": reranker.name if reranker else None,
        "rerank_depth": rerank_depth if reranker else None,
    }
    if rerank_latencies:
        rerank_ms = np.array(rerank_latencies) * 1000
        row["rerank_p50_ms"] = round(float(np.percentile(rerank_ms, 50)), 2)
        row["rerank_p99_ms"] = round(float(np.percentile(rerank_ms, 99)), 2)
    if per_query:  # for paired significance tests (vector/eval/compare.py)
        row["per_query_ndcg@10"] = {q: round(v, 4) for q, v in per_query_ndcg(results, dataset.qrels).items()}
    return row


def run_experiment(
    dataset: BeirDataset,
    chunker: Chunker,
    embedder: Embedder,
    k_chunks: int = 100,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    index_spec: str = "flat",
) -> dict:
    prepared = prepare(dataset, chunker, embedder, cache_dir)
    return evaluate_index(dataset, chunker, embedder, prepared, index_spec, k_chunks)


def git_info() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        # Only tracked files count: untracked ones (like the results file being
        # appended to) can't change what the code computes.
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
        return {"commit": git("rev-parse", "--short", "HEAD"), "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def format_row(row: dict) -> str:
    m = row["metrics"]
    return (
        f"{row['model'].split('/')[-1]:<24} {row['chunker']:<18} {row['index']:<16} "
        f"{(row['rerank'] or '-').split('/')[-1][:20]:<20} {row['n_chunks']:>6} "
        f"{row['truncated_frac']:>6.1%} {m['recall@1']:>6.3f} {m['recall@5']:>6.3f} "
        f"{m['recall@10']:>6.3f} {m['mrr@10']:>6.3f} {m['ndcg@10']:>6.3f} "
        f"{row['ann_recall@10']:>6.3f} {row['scored_mean']:>7.0f} "
        f"{row['search_p50_ms']:>7.3f} {row['search_p99_ms']:>7.3f}"
    )


HEADER = (
    f"{'model':<24} {'chunker':<18} {'index':<16} {'rerank':<20} {'chunks':>6} {'trunc':>6} {'R@1':>6} {'R@5':>6} "
    f"{'R@10':>6} {'MRR':>6} {'nDCG':>6} {'ANN@10':>6} {'scored':>7} {'p50ms':>7} {'p99ms':>7}"
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="minilm")
    parser.add_argument("--chunker", nargs="+", default=["whole"], help="e.g. whole fixed-200-40")
    parser.add_argument("--index", nargs="+", default=["flat"], help="e.g. flat hnsw-16-200-64")
    parser.add_argument("--k-chunks", type=int, default=100)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-save", action="store_true", help="print results without recording them")
    parser.add_argument("--per-query", action="store_true", help="also record each query's nDCG@10")
    parser.add_argument("--rerank", nargs="+", default=["none"], choices=["none", *RERANKERS],
                        help="cross-encoder(s) to re-order the top chunks with")
    parser.add_argument("--rerank-depth", type=int, default=50, help="how many top chunks to re-rank")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # model-hub request noise
    chunkers = [parse_chunker(spec) for spec in args.chunker]  # fail fast on typos
    for spec in args.index:
        parse_index(spec, dim=1)
    dataset = load_scifact()
    embedder = MODELS[args.model]()
    rerankers = {name: None if name == "none" else _load_reranker(name) for name in args.rerank}
    info = git_info()

    print("\n" + HEADER)  # rows print as they finish; HNSW builds can take minutes
    for chunker in chunkers:
        prepared = prepare(dataset, chunker, embedder)  # embedded once, shared by every index
        built: dict = {}
        for spec in args.index:
            for rr_name in args.rerank:
                reranker = rerankers[rr_name]
                row = evaluate_index(
                    dataset, chunker, embedder, prepared, spec, args.k_chunks, built,
                    args.per_query, reranker, args.rerank_depth,
                )
                row.update(info, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
                print(format_row(row), flush=True)
                if not args.no_save:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    with open(args.out, "a") as f:
                        f.write(json.dumps(row) + "\n")

    if info["dirty"]:
        print("\nwarning: uncommitted changes; commit before recording numbers you plan to publish")


if __name__ == "__main__":
    main()
