"""Speed benchmark on SIFT1M: exact search vs our HNSW vs FAISS HNSW, at growing sizes.

    python -m vector.eval.bench --n 10000 100000 \\
        --index flat hnsw-16-100-32 hnsw-16-100-64 faiss-hnsw-16-100-32 faiss-hnsw-16-100-64

For each base size n: exact search over the first n SIFT vectors is the
answer key; every index is built (graphs reused across ef_search), then each
query is timed. One JSON row per (n, index) goes to --out.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from vector.data.sift import load_sift
from vector.eval.run import REPO_ROOT, build_index, git_info, parse_index
from vector.index.base import normalize

DEFAULT_OUT = REPO_ROOT / "results" / "phase4.jsonl"
log = logging.getLogger(__name__)


def exact_top_k(base: np.ndarray, queries: np.ndarray, k: int, batch: int = 256) -> np.ndarray:
    """True top-k ids by cosine, in batches so the score matrix stays small (256 × n floats)."""
    base, queries = normalize(base), normalize(queries)
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), batch):
        scores = queries[start : start + batch] @ base.T
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        out[start : start + batch] = np.take_along_axis(top, order, axis=1)
    return out


def benchmark(index, queries: np.ndarray, truth: np.ndarray, k: int) -> dict:
    index.search(queries[0], k)  # untimed warm-up
    latencies, hits, scored = [], 0, []
    for q, true_ids in zip(queries, truth):
        start = time.perf_counter()
        ids, _ = index.search(q, k)
        latencies.append(time.perf_counter() - start)
        hits += len(set(ids.tolist()) & set(true_ids.tolist()))
        scored.append(getattr(index, "scored_last_search", None))
    ms = np.array(latencies) * 1000
    return {
        f"recall@{k}": round(hits / truth.size, 4),
        "search_p50_ms": round(float(np.percentile(ms, 50)), 4),
        "search_p99_ms": round(float(np.percentile(ms, 99)), 4),
        "qps": round(len(queries) / float(np.sum(latencies)), 1),  # one query at a time
        "scored_mean": None if scored[0] is None else round(float(np.mean(scored)), 1),
    }


HEADER = f"{'n':>9} {'index':<22} {'recall':>7} {'p50ms':>8} {'p99ms':>8} {'QPS':>9} {'scored':>8} {'build s':>8}"


def format_row(row: dict, k: int) -> str:
    scored = "-" if row["scored_mean"] is None else f"{row['scored_mean']:.0f}"
    return (
        f"{row['n']:>9,} {row['index']:<22} {row[f'recall@{k}']:>7.3f} {row['search_p50_ms']:>8.3f} "
        f"{row['search_p99_ms']:>8.3f} {row['qps']:>9,.0f} {scored:>8} {row['build_seconds']:>8.1f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, nargs="+", default=[10_000], help="base sizes, e.g. 10000 100000")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--index", nargs="+", default=["flat", "hnsw-16-100-64", "faiss-hnsw-16-100-64"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for spec in args.index:
        parse_index(spec, dim=1)  # fail fast on typos
    info = git_info()

    print(HEADER, flush=True)
    for n in args.n:
        base, queries = load_sift(n, args.queries)
        truth = exact_top_k(base, queries, args.k)
        built: dict = {}
        for spec in args.index:
            log.info("n=%d %s: building", n, spec)
            index, build_seconds = build_index(spec, base, built)
            row = {
                "dataset": "sift1m", "n": n, "dim": base.shape[1], "n_queries": len(queries),
                "k": args.k, "index": spec, "build_seconds": round(build_seconds, 3),
                **benchmark(index, queries, truth, args.k),
                **info, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            print(format_row(row, args.k), flush=True)
            if not args.no_save:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
    if info["dirty"]:
        print("\nwarning: uncommitted changes; commit before recording numbers you plan to publish")


if __name__ == "__main__":
    main()
