import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vector.chunking import WholeDocumentChunker
from vector.data.beir import Document
from vector.index.hnsw import HNSWIndex
from vector.service.app import create_app
from vector.service.ratelimit import TokenBucketLimiter
from vector.service.store import RWLock, SearchStore

DOCS = [
    {"id": "d1", "title": "Fruit", "text": "apple banana"},
    {"id": "d2", "title": "Berry", "text": "cherry"},
    {"id": "d3", "title": "Mix", "text": "banana banana cherry date"},
]


@pytest.fixture
def store(bow):
    return SearchStore(bow, WholeDocumentChunker(), HNSWIndex(bow.dim, M=4, ef_construction=16))


@pytest.fixture
def client(store):
    return TestClient(create_app(store))


def test_add_then_query(client):
    r = client.post("/documents", json={"documents": DOCS})
    assert r.status_code == 201 and r.json()["added_chunks"] == 3 and r.json()["documents"] == 3
    r = client.post("/query", json={"query": "cherry", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert [h["doc_id"] for h in body["results"]] == ["d2", "d3"]
    assert body["results"][0]["title"] == "Berry" and body["results"][0]["score"] == pytest.approx(1.0)
    assert set(body["timings_ms"]) == {"embed_ms", "search_ms"}


def test_documents_already_present_are_skipped(client):
    client.post("/documents", json={"documents": DOCS})
    r = client.post("/documents", json={"documents": DOCS[:1] + [{"id": "d4", "text": "date"}]})
    assert r.json()["added_chunks"] == 1 and r.json()["documents"] == 4


def test_bad_requests_are_rejected(client):
    assert client.post("/query", json={"query": ""}).status_code == 422
    assert client.post("/query", json={"query": "x", "k": 0}).status_code == 422
    assert client.post("/documents", json={"documents": []}).status_code == 422
    # Re-ranking was asked for, but this server has no re-ranker.
    assert client.post("/query", json={"query": "apple", "rerank": True}).status_code == 400


def test_health(client):
    client.post("/documents", json={"documents": DOCS})
    assert client.get("/health").json() == {"status": "ok", "documents": 3, "chunks": 3}


def test_save_and_load_round_trip(store, bow, tmp_path):
    store.add_documents([Document(d["id"], d["title"], d["text"]) for d in DOCS])
    before, _ = store.query("banana", k=3)
    store.save(tmp_path / "idx")
    store.save(tmp_path / "idx")  # saving over an existing save works too

    loaded = SearchStore.load(tmp_path / "idx", bow, WholeDocumentChunker())
    after, _ = loaded.query("banana", k=3)
    assert [h.doc_id for h in after] == [h.doc_id for h in before]
    stats = loaded.stats()
    assert (stats["documents"], stats["chunks"]) == (3, 3)
    # The loaded index keeps working as a live index: new documents get linked in.
    loaded.add_documents([Document("d4", "", "apple apple")])
    assert loaded.query("apple", k=1)[0][0].doc_id in {"d1", "d4"}


def test_load_refuses_a_different_model(store, tmp_path):
    from conftest import BagOfWordsEmbedder

    store.save(tmp_path / "idx")
    other = BagOfWordsEmbedder(["apple"], name="other-model")
    with pytest.raises(ValueError):
        SearchStore.load(tmp_path / "idx", other, WholeDocumentChunker())


class ShortestMatchReranker:
    name = "fake/shortest-match"

    def score(self, query, passages):
        return np.array([(query in p.split()) / len(p.split()) for p in passages])


def test_rerank_endpoint(bow):
    store = SearchStore(bow, WholeDocumentChunker(), reranker=ShortestMatchReranker())
    client = TestClient(create_app(store))
    client.post("/documents", json={"documents": DOCS})
    # Search alone ranks d3 ("banana banana ...") first; the re-ranker prefers the shorter d1.
    plain = client.post("/query", json={"query": "banana", "k": 2}).json()["results"]
    rr = client.post("/query", json={"query": "banana", "k": 2, "rerank": True}).json()
    assert plain[0]["doc_id"] == "d3" and rr["results"][0]["doc_id"] == "d1"
    assert "rerank_ms" in rr["timings_ms"]


def test_rate_limit_returns_429_with_retry_after(store):
    now = [0.0]
    limiter = TokenBucketLimiter(rate=1, capacity=2, clock=lambda: now[0])
    client = TestClient(create_app(store, limiter))
    headers = {"x-api-key": "alice"}
    assert client.post("/query", json={"query": "a"}, headers=headers).status_code == 200
    assert client.post("/query", json={"query": "a"}, headers=headers).status_code == 200
    r = client.post("/query", json={"query": "a"}, headers=headers)
    assert r.status_code == 429 and r.headers["retry-after"] == "1"
    # Another client has its own bucket.
    assert client.post("/query", json={"query": "a"}, headers={"x-api-key": "bob"}).status_code == 200
    now[0] = 1.0  # one second later, one token is back
    assert client.post("/query", json={"query": "a"}, headers=headers).status_code == 200


def test_token_bucket_math():
    now = [0.0]
    bucket = TokenBucketLimiter(rate=2, capacity=4, clock=lambda: now[0])
    assert bucket.allow("c", 3).allowed  # 4 → 1
    d = bucket.allow("c", 3)
    assert not d.allowed and d.retry_after == pytest.approx(1.0)  # need 2 more at 2/s
    now[0] = 1.0
    assert bucket.allow("c", 3).allowed  # refilled to 3 → 0
    now[0] = 100.0
    assert bucket.allow("c", 4).remaining == 0  # refill caps at capacity
    assert not bucket.allow("c", 5).allowed  # more than capacity: never
    with pytest.raises(ValueError):
        TokenBucketLimiter(rate=0, capacity=1)


def test_concurrent_queries_while_adding(bow):
    """Readers and a writer hammer the store together; nothing crashes and every result is valid."""
    store = SearchStore(bow, WholeDocumentChunker(), HNSWIndex(bow.dim, M=4, ef_construction=16))
    store.add_documents([Document("seed", "", "apple")])
    errors = []

    def reader():
        try:
            for _ in range(200):
                hits, _ = store.query("banana cherry", k=5)
                assert all(h.doc_id in store.titles for h in hits)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def writer():
        words = ["apple", "banana", "cherry", "date"]
        for i in range(200):
            store.add_documents([Document(f"w{i}", "", f"{words[i % 4]} {words[(i * 3) % 4]}")])

    threads = [threading.Thread(target=reader) for _ in range(4)] + [threading.Thread(target=writer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and store.stats()["chunks"] == 201


def test_rwlock_writer_excludes_readers():
    lock, log = RWLock(), []
    with lock.read():
        with lock.read():  # two readers at once is fine
            log.append("two readers")
    with lock.write():
        log.append("writer")
    assert log == ["two readers", "writer"]


def test_answer_endpoint_cites_retrieved_documents(client, store):
    from test_generate import FakeClient
    from vector.generate import OpenAIAnswerer

    fake = FakeClient("Cherry is a berry [1]. Nothing about [9].")
    app_client = TestClient(create_app(store, answerer=OpenAIAnswerer(model="m", client=fake)))
    app_client.post("/documents", json={"documents": DOCS})
    body = app_client.post("/answer", json={"question": "cherry", "k": 2}).json()
    assert body["citations"] == [{"doc_id": "d2", "title": "Berry"}]  # [1] = top hit d2
    assert body["invalid_citations"] == [9]
    assert [s["doc_id"] for s in body["sources"]] == ["d2", "d3"]
    assert "[1] Berry" in fake.requests[0]["messages"][1]["content"]
    # Without an answerer configured, the endpoint says so instead of failing.
    assert client.post("/answer", json={"question": "cherry"}).status_code == 400
