"""Run the search API.

    python -m vector.service                       # bge-small + whole docs, SciFact preloaded
    python -m vector.service --rerank --rate 20 --burst 40 --port 8000

On first start it builds an HNSW index over SciFact and saves it under
--data-dir; later starts load the saved index in seconds instead of rebuilding.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import uvicorn

from vector.data.beir import DEFAULT_DATA_DIR, load_scifact
from vector.eval.run import MODELS, parse_chunker
from vector.index.hnsw import HNSWIndex
from vector.rerank import RERANKERS, CrossEncoderReranker
from vector.service.app import create_app
from vector.service.ratelimit import TokenBucketLimiter
from vector.service.store import SearchStore

log = logging.getLogger("vector.service")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="bge-small")
    parser.add_argument("--chunker", default="whole")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR / "service")
    parser.add_argument("--no-preload", action="store_true", help="start empty instead of loading SciFact")
    parser.add_argument("--rerank", action="store_true", help="enable cross-encoder re-ranking")
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--answer", action="store_true", help="enable /answer (needs OPENAI_API_KEY in .env)")
    parser.add_argument("--rate", type=float, help="rate limit: tokens per second per client")
    parser.add_argument("--burst", type=float, help="rate limit: bucket capacity (default 2 x rate)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers-threads", type=int, default=8, help="thread pool size for requests")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    embedder = MODELS[args.model]()
    chunker = parse_chunker(args.chunker)
    reranker = CrossEncoderReranker(RERANKERS["ms-marco-minilm"]) if args.rerank else None
    save_dir = args.data_dir / f"{args.model}-{chunker.name}"

    start = time.perf_counter()
    if (save_dir / "meta.json").exists():
        store = SearchStore.load(save_dir, embedder, chunker, reranker=reranker)
        log.info("loaded %d chunks from %s in %.1fs", len(store.chunks), save_dir, time.perf_counter() - start)
    else:
        store = SearchStore(embedder, chunker, HNSWIndex(embedder.dim, M=16, ef_construction=100), reranker)
        if not args.no_preload:
            log.info("building index over SciFact (first start only)...")
            store.add_documents(list(load_scifact().corpus.values()))
            store.save(save_dir)
            log.info("built and saved %d chunks in %.1fs", len(store.chunks), time.perf_counter() - start)
    store.index.ef_search = args.ef_search

    limiter = None
    if args.rate:
        limiter = TokenBucketLimiter(args.rate, args.burst or 2 * args.rate)
    answerer = None
    if args.answer:
        from vector.generate import OpenAIAnswerer

        answerer = OpenAIAnswerer()
    app = create_app(store, limiter, save_dir, answerer)

    import anyio.to_thread  # FastAPI's thread pool for plain-def endpoints

    @app.on_event("startup")
    async def _size_thread_pool() -> None:
        anyio.to_thread.current_default_thread_limiter().total_tokens = args.workers_threads

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
