import numpy as np
import pytest

from vector.chunking import WholeDocumentChunker
from vector.eval.compare import matches
from vector.eval.run import evaluate_index, prepare
from vector.rerank import rerank


def test_rerank_reorders_only_the_head():
    ids = [10, 11, 12, 13, 14]
    # Scores for the top 3 only: 12 is best, then 10, then 11. 13 and 14 keep their places.
    assert rerank(ids, np.array([0.5, 0.1, 0.9]), depth=3) == [12, 10, 11, 13, 14]
    assert rerank(ids, np.array([]), depth=0) == ids
    # Ties keep the original (search) order.
    assert rerank(ids, np.array([1.0, 1.0]), depth=2) == ids


class ShortestMatchReranker:
    """Fake cross-encoder that, like the real one, judges each passage against the query:
    passages mentioning the query word score higher the shorter (more focused) they are."""

    name = "fake/shortest-match"

    def __init__(self):
        self.queries = []

    def score(self, query, passages):
        self.queries.append(query)
        return np.array([(query in p.split()) / len(p.split()) for p in passages])


def test_reranker_fixes_a_ranking_in_the_pipeline(dataset, bow, tmp_path):
    # From conftest: q3 ("banana") has answer d1 ("apple banana"), but search ranks
    # d3 ("banana banana cherry date") first. The re-ranker prefers the shorter match, d1.
    prepared = prepare(dataset, WholeDocumentChunker(), bow, tmp_path)
    plain = evaluate_index(dataset, WholeDocumentChunker(), bow, prepared, k_chunks=10)
    fake = ShortestMatchReranker()
    reranked = evaluate_index(
        dataset, WholeDocumentChunker(), bow, prepared, k_chunks=10, reranker=fake, rerank_depth=10
    )
    assert plain["metrics"]["mrr@10"] == pytest.approx((1 + 1 + 0.5) / 3, abs=1e-4)
    assert reranked["metrics"]["mrr@10"] == pytest.approx(1.0)  # every answer now at rank 1
    assert reranked["rerank"] == "fake/shortest-match" and reranked["rerank_depth"] == 10
    assert "rerank_p50_ms" in reranked and plain["rerank"] is None
    assert fake.queries == ["apple", "cherry", "banana"]  # raw query text, no embedding prefix


def test_shallow_rerank_depth_cannot_reach_lower_results(dataset, bow, tmp_path):
    # With depth 1 only the top hit is "re-ranked" (alone), so nothing moves.
    prepared = prepare(dataset, WholeDocumentChunker(), bow, tmp_path)
    shallow = evaluate_index(
        dataset, WholeDocumentChunker(), bow, prepared, k_chunks=10,
        reranker=ShortestMatchReranker(), rerank_depth=1,
    )
    assert shallow["metrics"]["mrr@10"] == pytest.approx((1 + 1 + 0.5) / 3, abs=1e-4)


def test_selector_can_tell_reranked_rows_apart():
    plain = {"model": "org/bge-small", "chunker": "whole", "index": "flat", "rerank": None}
    rr = dict(plain, rerank="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert matches(plain, "bge-small:whole:flat:none") and not matches(rr, "bge-small:whole:flat:none")
    assert matches(rr, "bge-small:whole:flat:ms-marco") and not matches(plain, "bge-small:whole:flat:ms-marco")
    assert matches(plain, "bge-small:whole") and matches(rr, "bge-small:whole")
