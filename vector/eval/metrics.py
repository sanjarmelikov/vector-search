"""Retrieval quality metrics, computed at the document level.

The index returns chunks, but BEIR judges documents. Chunk hits are collapsed
to their parent document, keeping each document's best (first) rank.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def dedupe_docs(chunk_doc_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ranked = []
    for doc_id in chunk_doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
    return ranked


def recall_at_k(ranked: list[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevance: Mapping[str, int], k: int) -> float:
    """Normalized discounted cumulative gain, with linear gains (as trec_eval and BEIR use).

    A hit at rank r is worth grade / log2(r + 1), so rank 1 counts fully and
    later ranks count less. Dividing by the best possible ordering's score
    maps the result into [0, 1].
    """
    dcg = sum(
        relevance.get(doc_id, 0) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    results: Mapping[str, list[str]],
    qrels: Mapping[str, Mapping[str, int]],
    ks: Iterable[int] = (1, 5, 10),
    mrr_k: int = 10,
    ndcg_k: int = 10,
) -> dict[str, float]:
    """Average recall@k, MRR@k and nDCG@k over every judged query.

    `results` maps query_id -> ranked doc ids (already deduped). Queries with
    judgments but no results count as zeros rather than being skipped.
    """
    ks = list(ks)
    totals = {f"recall@{k}": 0.0 for k in ks}
    totals[f"mrr@{mrr_k}"] = 0.0
    totals[f"ndcg@{ndcg_k}"] = 0.0
    for query_id, relevant in qrels.items():
        ranked = results.get(query_id, [])
        for k in ks:
            totals[f"recall@{k}"] += recall_at_k(ranked, relevant, k)
        totals[f"mrr@{mrr_k}"] += reciprocal_rank(ranked, relevant, mrr_k)
        totals[f"ndcg@{ndcg_k}"] += ndcg_at_k(ranked, relevant, ndcg_k)
    n = max(len(qrels), 1)
    return {name: total / n for name, total in totals.items()}


def per_query_ndcg(
    results: Mapping[str, list[str]], qrels: Mapping[str, Mapping[str, int]], k: int = 10
) -> dict[str, float]:
    """nDCG@k for each judged query, for paired significance tests between systems."""
    return {query_id: ndcg_at_k(results.get(query_id, []), rel, k) for query_id, rel in qrels.items()}
