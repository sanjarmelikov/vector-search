import pytest

import math

from vector.eval.metrics import dedupe_docs, evaluate, ndcg_at_k, recall_at_k, reciprocal_rank


def test_dedupe_keeps_first_rank():
    assert dedupe_docs(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_recall_at_k():
    ranked = ["x", "a", "y", "b"]
    assert recall_at_k(ranked, {"a", "b"}, 1) == 0.0
    assert recall_at_k(ranked, {"a", "b"}, 2) == 0.5
    assert recall_at_k(ranked, {"a", "b"}, 4) == 1.0
    assert recall_at_k(ranked, set(), 4) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a"], {"a"}, 10) == 0.5
    assert reciprocal_rank(["x", "a"], {"a"}, 1) == 0.0
    assert reciprocal_rank([], {"a"}, 10) == 0.0


def test_evaluate_counts_missing_queries_as_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    scores = evaluate({"q1": ["a", "z"]}, qrels, ks=(1,), mrr_k=10)
    assert scores == {
        "recall@1": pytest.approx(0.5),
        "mrr@10": pytest.approx(0.5),
        "ndcg@10": pytest.approx(0.5),
    }


def test_ndcg_at_k():
    # Perfect ordering scores 1.
    assert ndcg_at_k(["a", "b", "x"], {"a": 1, "b": 1}, 10) == pytest.approx(1.0)
    # One relevant doc at rank 2: 1/log2(3) over an ideal of 1/log2(2) = 1.
    assert ndcg_at_k(["x", "a"], {"a": 1}, 10) == pytest.approx(1 / math.log2(3))
    # Higher grades should come first: ranking the grade-2 doc second is penalized.
    swapped = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10)
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert swapped == pytest.approx((1 + 2 / math.log2(3)) / ideal)
    # Hits past k don't count; no relevant docs scores 0.
    assert ndcg_at_k(["x", "a"], {"a": 1}, 1) == 0.0
    assert ndcg_at_k(["a"], {}, 10) == 0.0
