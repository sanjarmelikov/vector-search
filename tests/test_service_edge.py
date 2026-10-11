"""Edge cases for the search service: odd inputs, limits, and crash recovery of saves."""

import shutil
import threading

import pytest
from fastapi.testclient import TestClient

from vector.chunking import WholeDocumentChunker
from vector.data.beir import Document
from vector.index.hnsw import HNSWIndex
from vector.service.app import create_app
from vector.service.store import SearchStore

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


# ---- queries ----

def test_query_on_an_empty_index(client):
    r = client.post("/query", json={"query": "anything"})
    assert r.status_code == 200 and r.json()["results"] == []


@pytest.mark.parametrize("query", ["🍎🍌", "ÁPPLE—ñ", "日本語のクエリ", "apple\x00banana", "' OR 1=1; --"])
def test_unusual_query_text_is_handled(client, query):
    client.post("/documents", json={"documents": DOCS})
    r = client.post("/query", json={"query": query})
    assert r.status_code == 200 and isinstance(r.json()["results"], list)


def test_whitespace_only_query_is_rejected(client):
    assert client.post("/query", json={"query": "   \n\t"}).status_code == 422


def test_k_larger_than_the_index(client):
    client.post("/documents", json={"documents": DOCS})
    assert len(client.post("/query", json={"query": "banana", "k": 100}).json()["results"]) == 3


@pytest.mark.parametrize("body", [
    {"query": "x" * 2001},          # too long
    {"query": "x", "k": 101},       # k over the limit
    {"query": "x", "k": "ten"},     # wrong type
    {"k": 5},                       # missing query
    [1, 2, 3],                      # not an object
])
def test_invalid_query_bodies(client, body):
    assert client.post("/query", json=body).status_code == 422


def test_malformed_json(client):
    r = client.post("/query", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


# ---- documents ----

def test_duplicate_ids_in_one_request_keep_one(client):
    r = client.post("/documents", json={"documents": [{"id": "x", "text": "apple"}, {"id": "x", "text": "cherry"}]})
    assert r.json()["documents"] == 1 and r.json()["added_chunks"] == 1


def test_document_limits(client):
    assert client.post("/documents", json={"documents": [{"id": "big", "text": "w " * 150_000}]}).status_code == 422
    too_many = [{"id": str(i), "text": "apple"} for i in range(1001)]
    assert client.post("/documents", json={"documents": too_many}).status_code == 422
    assert client.post("/documents", json={"documents": [{"id": "", "text": "x"}]}).status_code == 422
    assert client.post("/documents", json={"documents": [{"id": "w", "text": "   "}]}).status_code == 422


def test_concurrent_adds_of_the_same_document(store):
    def add():
        store.add_documents([Document("same", "", "apple banana")])

    threads = [threading.Thread(target=add) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.stats()["documents"] == 1 and store.stats()["chunks"] == 1


# ---- crash recovery of saves ----

def docs():
    return [Document(d["id"], d["title"], d["text"]) for d in DOCS]


def test_crash_between_the_two_renames_is_recovered(store, bow, tmp_path):
    store.add_documents(docs())
    store.save(tmp_path / "idx")
    # Simulate dying mid-swap: live dir moved aside, new save not yet moved in.
    (tmp_path / "idx").rename(tmp_path / "idx.old")
    shutil.copytree(tmp_path / "idx.old", tmp_path / "idx.tmp")
    loaded = SearchStore.load(tmp_path / "idx", bow, WholeDocumentChunker())
    assert loaded.stats()["documents"] == 3  # the complete new save was used, not a rebuild
    assert (tmp_path / "idx").exists() and not (tmp_path / "idx.tmp").exists()


def test_crash_leaving_only_the_old_save_is_recovered(store, bow, tmp_path):
    store.add_documents(docs())
    store.save(tmp_path / "idx")
    (tmp_path / "idx").rename(tmp_path / "idx.old")  # died right after the first rename
    loaded = SearchStore.load(tmp_path / "idx", bow, WholeDocumentChunker())
    assert loaded.stats()["documents"] == 3


def test_save_after_a_crash_left_debris(store, tmp_path):
    store.add_documents(docs())
    (tmp_path / "idx.old").mkdir()
    (tmp_path / "idx.old" / "junk").write_text("x")
    (tmp_path / "idx.tmp").mkdir()
    (tmp_path / "idx.tmp" / "half-written").write_text("x")
    store.save(tmp_path / "idx")  # must not fail on leftovers
    store.save(tmp_path / "idx")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["idx"]


def test_corrupt_save_fails_loudly(store, bow, tmp_path):
    store.add_documents(docs())
    store.save(tmp_path / "idx")
    (tmp_path / "idx" / "meta.json").write_text("{truncated")
    with pytest.raises(ValueError):
        SearchStore.load(tmp_path / "idx", bow, WholeDocumentChunker())
