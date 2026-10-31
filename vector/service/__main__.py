"""Run the search API.

    python -m vector.service                       # bge-small + whole docs, SciFact preloaded
    python -m vector.service --rerank --rate 20 --burst 40 --port 8000

On first start it builds an HNSW index over SciFact and saves it under
--data-dir; later starts load the saved index in seconds instead of rebuilding.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from pathlib import Path

import uvicorn

from vector.data.beir import DEFAULT_DATA_DIR, load_scifact
from vector.eval.run import MODELS, parse_chunker
from vector.index.hnsw import HNSWIndex
from vector.rerank import RERANKERS, CrossEncoderReranker
from vector.service.app import create_app
from vector.service.ratelimit import TokenBucketLimiter
from vector.service.store import SearchStore, recover_possible

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
    parser.add_argument("--read-only", action="store_true", help="public demo: search only, POST /documents is refused")
    parser.add_argument("--trust-proxy", action="store_true",
                        help="take the client IP from X-Forwarded-For (only behind a reverse proxy that sets it)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers-threads", type=int, default=64, help="thread pool size for requests")
    parser.add_argument("--no-batching", action="store_true", help="embed each query alone (for comparison)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    embedder = MODELS[args.model]()
    chunker = parse_chunker(args.chunker)
    reranker = CrossEncoderReranker(RERANKERS["ms-marco-minilm"]) if args.rerank else None
    save_dir = args.data_dir / f"{args.model}-{chunker.name}"

    start = time.perf_counter()
    if (save_dir / "meta.json").exists() or recover_possible(save_dir):
        store = SearchStore.load(save_dir, embedder, chunker, reranker=reranker, batch_queries=not args.no_batching)
        log.info("loaded %d chunks from %s in %.1fs", len(store.chunks), save_dir, time.perf_counter() - start)
    else:
        store = SearchStore(embedder, chunker, HNSWIndex(embedder.dim, M=16, ef_construction=100), reranker,
                            batch_queries=not args.no_batching)
        if not args.no_preload:
            # Build in the background so the server is reachable (and reports "building")
            # right away, instead of looking dead to health checks for minutes.
            docs = list(load_scifact().corpus.values())
            log.info("building index over %d SciFact documents in the background...", len(docs))

            def build() -> None:
                store.build(docs, save_dir)
                log.info("built and saved %d chunks in %.1fs", len(store.chunks), time.perf_counter() - start)

            threading.Thread(target=build, name="index-build", daemon=True).start()
    store.index.ef_search = args.ef_search

    limiter = None
    if args.rate:
        limiter = TokenBucketLimiter(args.rate, args.burst or 2 * args.rate)
    answerer = None
    if args.answer:
        from vector.generate import OpenAIAnswerer

        answerer = OpenAIAnswerer()
    # Comma-separated keys from the environment (never on the command line, where `ps` shows them).
    api_keys = {k.strip() for k in os.environ.get("VECTOR_API_KEYS", "").split(",") if k.strip()} or None
    if api_keys is None and not args.read_only and args.host not in ("127.0.0.1", "localhost"):
        log.warning("listening on %s with no API keys: anyone who can reach it can add documents. "
                    "Set VECTOR_API_KEYS.", args.host)
    app = create_app(store, limiter, save_dir, answerer, threads=args.workers_threads, api_keys=api_keys,
                     read_only=args.read_only)
    # Behind a proxy every connection comes from the proxy's address; without this the
    # per-IP rate limit would treat all visitors as one client.
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                proxy_headers=args.trust_proxy, forwarded_allow_ips="*" if args.trust_proxy else None)


if __name__ == "__main__":
    main()
