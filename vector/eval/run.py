"""Run one retrieval experiment end to end and record the results.

    python -m vector.eval.run --model minilm --chunker whole fixed-200-40

Each (model, chunker) pair becomes one JSON row in the --out file
(default results/phase1.jsonl), tagged with the git commit that produced it.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from vector.chunking import (
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
from vector.eval.metrics import dedupe_docs, evaluate
from vector.index.flat import FlatIndex

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results" / "phase1.jsonl"


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


def run_experiment(
    dataset: BeirDataset,
    chunker: Chunker,
    embedder: Embedder,
    k_chunks: int = 100,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict:
    chunks = chunk_documents(list(dataset.corpus.values()), chunker)
    texts = [c.text for c in chunks]

    # Truncation: how much of the corpus the model never actually reads.
    tokens = np.array(embedder.count_tokens(texts))
    truncated = float(np.mean(tokens > embedder.max_tokens))

    chunk_vectors = cached_embed(embedder, texts, cache_dir)
    query_ids = list(dataset.queries)
    query_texts = [embedder.query_prefix + dataset.queries[q] for q in query_ids]
    query_vectors = cached_embed(embedder, query_texts, cache_dir)

    start = time.perf_counter()
    index = FlatIndex(embedder.dim)
    index.add(chunk_vectors)
    build_seconds = time.perf_counter() - start

    # One untimed search first, so one-off setup costs don't land in the latency numbers.
    index.search(query_vectors[0], k_chunks)

    results: dict[str, list[str]] = {}
    latencies = []
    for query_id, vector in zip(query_ids, query_vectors):
        start = time.perf_counter()
        ids, _ = index.search(vector, k_chunks)
        latencies.append(time.perf_counter() - start)
        # Several chunks can share a document; keep each doc's best rank.
        results[query_id] = dedupe_docs(chunks[i].doc_id for i in ids)

    latencies_ms = np.array(latencies) * 1000
    return {
        "model": embedder.name,
        "chunker": chunker.name,
        "n_docs": len(dataset.corpus),
        "n_chunks": len(chunks),
        "n_queries": len(query_ids),
        "dim": embedder.dim,
        "k_chunks": k_chunks,
        "tokens_mean": round(float(tokens.mean()), 1),
        "tokens_max": int(tokens.max()),
        "max_tokens": embedder.max_tokens,
        "query_prefix": embedder.query_prefix,
        "truncated_frac": round(truncated, 4),
        "metrics": {name: round(value, 4) for name, value in evaluate(results, dataset.qrels).items()},
        "search_p50_ms": round(float(np.percentile(latencies_ms, 50)), 4),
        "search_p99_ms": round(float(np.percentile(latencies_ms, 99)), 4),
        "build_seconds": round(build_seconds, 4),
    }


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
        f"{row['model'].split('/')[-1]:<24} {row['chunker']:<18} {row['n_chunks']:>6} "
        f"{row['truncated_frac']:>6.1%} {m['recall@1']:>6.3f} {m['recall@5']:>6.3f} "
        f"{m['recall@10']:>6.3f} {m['mrr@10']:>6.3f} {m['ndcg@10']:>6.3f} "
        f"{row['search_p50_ms']:>7.3f} {row['search_p99_ms']:>7.3f}"
    )


HEADER = (
    f"{'model':<24} {'chunker':<18} {'chunks':>6} {'trunc':>6} {'R@1':>6} {'R@5':>6} "
    f"{'R@10':>6} {'MRR':>6} {'nDCG':>6} {'p50ms':>7} {'p99ms':>7}"
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="minilm")
    parser.add_argument("--chunker", nargs="+", default=["whole"], help="e.g. whole fixed-200-40")
    parser.add_argument("--k-chunks", type=int, default=100)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-save", action="store_true", help="print results without recording them")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # model-hub request noise
    chunkers = [parse_chunker(spec) for spec in args.chunker]  # fail fast on typos
    dataset = load_scifact()
    embedder = MODELS[args.model]()
    info = git_info()

    rows = []
    for chunker in chunkers:
        row = run_experiment(dataset, chunker, embedder, args.k_chunks)
        row.update(info, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        rows.append(row)
        if not args.no_save:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")

    print("\n" + HEADER)
    for row in rows:
        print(format_row(row))
    if info["dirty"]:
        print("\nwarning: uncommitted changes; commit before recording numbers you plan to publish")


if __name__ == "__main__":
    main()
