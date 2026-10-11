import pytest

from vector.chunking import FixedSizeChunker, WholeDocumentChunker
from vector.data.beir import BeirDataset, Document
from vector.eval.run import parse_chunker, run_experiment


@pytest.fixture
def dataset():
    corpus = {
        "d1": Document("d1", "", "apple banana"),
        "d2": Document("d2", "", "cherry"),
        "d3": Document("d3", "", "banana banana cherry date"),
    }
    queries = {"q1": "apple", "q2": "cherry", "q3": "banana"}
    # q3's labeled answer is d1, but d3 mentions banana more, so d1 lands at rank 2.
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d1": 1}}
    return BeirDataset(corpus, queries, qrels)


def test_whole_documents(dataset, bow, tmp_path):
    row = run_experiment(dataset, WholeDocumentChunker(), bow, k_chunks=10, cache_dir=tmp_path)
    assert row["n_chunks"] == 3 and row["n_queries"] == 3
    m = row["metrics"]
    assert m["recall@1"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall@5"] == pytest.approx(1.0)
    assert m["mrr@10"] == pytest.approx((1 + 1 + 0.5) / 3, abs=1e-4)
    # Only d3 (4 words) exceeds the fake model's 3-token limit.
    assert row["truncated_frac"] == pytest.approx(1 / 3, abs=1e-4)


def test_chunks_map_back_to_documents(dataset, bow, tmp_path):
    # d3 splits into "banana banana" and "cherry date": two chunks, one document.
    row = run_experiment(dataset, FixedSizeChunker(2, 0), bow, k_chunks=10, cache_dir=tmp_path)
    assert row["n_chunks"] == 4
    # Index ids point into the chunk list; each hit must map back to its parent doc.
    # q3's best chunk is d3#0 ("banana banana"), so labeled doc d1 is still rank 2.
    assert row["metrics"]["mrr@10"] == pytest.approx((1 + 1 + 0.5) / 3, abs=1e-4)


def test_parse_chunker():
    assert parse_chunker("whole").name == "whole"
    assert parse_chunker("fixed-200-40").name == "fixed-200-40"
    assert parse_chunker("recursive-100-0").name == "recursive-100-0"
    for bad in ("fixed", "fixed-200", "semantic-200-40", "fixed-a-b"):
        with pytest.raises(ValueError):
            parse_chunker(bad)
