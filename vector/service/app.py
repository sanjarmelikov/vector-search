"""HTTP API over a SearchStore.

Endpoints are plain `def` (not `async def`): search and embedding are CPU work,
and FastAPI runs plain functions in a thread pool, so one slow request doesn't
block the server's event loop. NumPy and PyTorch release the GIL inside their
C code, so threads do overlap in the heavy parts.
"""

from __future__ import annotations

import hmac
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio.to_thread

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from vector.generate import OpenAIAnswerer, Source
from vector.service.ratelimit import RateLimiter
from vector.service.store import SearchStore, as_dicts, documents_from_payload


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace text")
    return value


class DocumentIn(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    title: str = Field(default="", max_length=1000)
    # Bounded so one request can't make the server embed (and hold) unlimited text.
    text: str = Field(min_length=1, max_length=200_000)

    _check_text = field_validator("id", "text")(_not_blank)


class DocumentsIn(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=1000)


class QueryIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=10, ge=1, le=100)
    rerank: bool = False

    _check_query = field_validator("query")(_not_blank)


class AnswerIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=10)
    rerank: bool = False

    _check_question = field_validator("question")(_not_blank)


# Cost of each endpoint in rate-limit tokens: a re-ranked query runs a second
# model over 50 passages, and an answer calls a paid LLM, so they cost more.
COSTS = {"query": 1, "query_rerank": 5, "documents": 10, "answer": 20}


def create_app(
    store: SearchStore,
    limiter: RateLimiter | None = None,
    save_dir: Path | None = None,
    answerer: OpenAIAnswerer | None = None,
    threads: int | None = None,
    api_keys: set[str] | None = None,
    max_body_bytes: int = 10 * 1024 * 1024,
) -> FastAPI:
    """Build the app.

    api_keys: if given, every endpoint except /health requires `x-api-key` to be one of
    them (401 otherwise), and rate limits are per verified key. Without keys the API is
    open (local use) and rate limits are per client IP, because an unverified key header
    could be changed on every request to dodge the limit.
    """
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if threads:  # size the thread pool that runs the plain-def endpoints
            anyio.to_thread.current_default_thread_limiter().total_tokens = threads
        yield

    app = FastAPI(title="vector-search", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # 1. Size: refuse oversized bodies from the header, before reading them.
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_body_bytes:
            return JSONResponse(status_code=413, content={"detail": "request body too large"})
        # 2. Authentication (everything except the health check).
        if api_keys and request.url.path != "/health":
            given = request.headers.get("x-api-key", "")
            # compare_digest takes the same time whether the first or last character differs
            if not any(hmac.compare_digest(given, k) for k in api_keys):
                return JSONResponse(status_code=401, content={"detail": "missing or invalid API key"},
                                    headers={"WWW-Authenticate": "ApiKey"})
        return await call_next(request)

    def client_id(request: Request) -> str:
        if api_keys:
            return "key:" + request.headers["x-api-key"]  # verified by the middleware
        return "ip:" + (request.client.host if request.client else "unknown")

    def charge(request: Request, cost: int) -> None:
        """Ask the rate limiter whether this client may spend `cost` tokens; 429 if not."""
        if limiter is None:
            return
        decision = limiter.allow(client_id(request), cost)
        if not decision.allowed:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(max(1, round(decision.retry_after)))},
            )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", **store.stats()}

    @app.post("/query")
    def query(body: QueryIn, request: Request) -> dict:
        charge(request, COSTS["query_rerank" if body.rerank else "query"])
        if body.rerank and store.reranker is None:
            raise HTTPException(status_code=400, detail="re-ranking is not enabled on this server")
        hits, timings = store.query(body.query, body.k, body.rerank)
        return {"query": body.query, "results": as_dicts(hits), "timings_ms": timings}

    @app.post("/answer")
    def answer(body: AnswerIn, request: Request) -> dict:
        if answerer is None:
            raise HTTPException(status_code=400, detail="answers are not enabled (start with --answer)")
        charge(request, COSTS["answer"])
        hits, timings = store.query(body.question, body.k, body.rerank and store.reranker is not None)
        sources = [Source(h.doc_id, h.title, h.text) for h in hits]
        start = time.perf_counter()
        result = answerer.answer(body.question, sources)
        timings["generate_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return {
            "question": body.question,
            "answer": result.text,
            "citations": [{"doc_id": s.doc_id, "title": s.title} for s in result.cited],
            "invalid_citations": result.invalid_citations,
            "sources": [{"n": i, "doc_id": h.doc_id, "title": h.title} for i, h in enumerate(hits, 1)],
            "model": result.model,
            "timings_ms": timings,
        }

    @app.post("/documents", status_code=201)
    def add_documents(body: DocumentsIn, request: Request) -> dict:
        charge(request, COSTS["documents"])
        added = store.add_documents(documents_from_payload([d.model_dump() for d in body.documents]))
        if save_dir is not None and added:
            store.save(save_dir)  # persist before acknowledging, so a crash can't lose the write
        return {"added_chunks": added, **store.stats()}

    @app.exception_handler(ValueError)
    def bad_value(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return app
