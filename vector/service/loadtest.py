"""Load test: many simultaneous clients against a running server, latency under pressure.

    python -m vector.service &                       # in another terminal
    python -m vector.service.loadtest --concurrency 1 4 16 64 --duration 20

Each of `concurrency` clients sends SciFact claims back to back for `duration`
seconds (a "closed loop": a client waits for its answer before sending the next).
We record every request's latency and status: p50/p99 latency, throughput, and
how many requests the rate limiter turned away (429).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

from vector.data.beir import load_scifact
from vector.eval.run import REPO_ROOT, git_info

DEFAULT_OUT = REPO_ROOT / "results" / "phase7.jsonl"


async def client_loop(http: httpx.AsyncClient, url: str, queries: list[str], offset: int,
                      deadline: float, body: dict, api_key: str, log: list) -> None:
    i = offset
    while time.perf_counter() < deadline:
        payload = {"query": queries[i % len(queries)], **body}
        start = time.perf_counter()
        try:
            r = await http.post(url, json=payload, headers={"x-api-key": api_key})
            status = r.status_code
        except httpx.HTTPError:
            status = -1  # connection error / timeout
        log.append((time.perf_counter() - start, status))
        i += 1


async def run_level(base_url: str, concurrency: int, duration: float, queries: list[str],
                    body: dict, shared_key: bool) -> dict:
    log: list[tuple[float, int]] = []
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, timeout=60, limits=limits) as http:
        deadline = time.perf_counter() + duration
        start = time.perf_counter()
        await asyncio.gather(*(
            client_loop(http, "/query", queries, c * 997, deadline, body,
                        "shared" if shared_key else f"client-{c}", log)
            for c in range(concurrency)
        ))
        elapsed = time.perf_counter() - start
    ok = np.array([lat for lat, status in log if status == 200]) * 1000
    statuses = [status for _, status in log]
    return {
        "concurrency": concurrency,
        "duration_s": round(elapsed, 2),
        "requests": len(log),
        "ok": int(np.sum(np.array(statuses) == 200)),
        "rate_limited": statuses.count(429),
        "errors": sum(1 for s in statuses if s not in (200, 429)),
        "throughput_rps": round(len(ok) / elapsed, 1),
        "p50_ms": round(float(np.percentile(ok, 50)), 2) if len(ok) else None,
        "p99_ms": round(float(np.percentile(ok, 99)), 2) if len(ok) else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--shared-key", action="store_true", help="all clients share one API key (one bucket)")
    parser.add_argument("--label", default="", help="free text saved with the rows, e.g. 'rate=50'")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    queries = list(load_scifact().queries.values())
    health = httpx.get(f"{args.url}/health", timeout=10).json()
    info = git_info()
    print(f"server: {health}")
    print(f"{'conc':>5} {'reqs':>7} {'ok':>7} {'429':>6} {'err':>5} {'rps':>8} {'p50 ms':>8} {'p99 ms':>8}")
    for c in args.concurrency:
        row = asyncio.run(run_level(args.url, c, args.duration, queries,
                                    {"k": 10, "rerank": args.rerank}, args.shared_key))
        row.update(label=args.label, rerank=args.rerank, shared_key=args.shared_key,
                   server=health, **info, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))
        print(f"{c:>5} {row['requests']:>7} {row['ok']:>7} {row['rate_limited']:>6} {row['errors']:>5} "
              f"{row['throughput_rps']:>8.1f} {row['p50_ms'] or 0:>8.1f} {row['p99_ms'] or 0:>8.1f}", flush=True)
        if not args.no_save:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
