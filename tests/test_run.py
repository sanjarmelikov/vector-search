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


def test_query_prefix_applies_to_queries_only(dataset, bow, tmp_path):
    # The prefix "banana banana " pulls every query toward d3. q2 ("cherry") now
    # ranks d3 > d1 > d2, so its answer drops to rank 3; documents are untouched.
    bow.query_prefix = "banana banana "
    row = run_experiment(dataset, WholeDocumentChunker(), bow, k_chunks=10, cache_dir=tmp_path)
    assert row["query_prefix"] == "banana banana "
    assert row["truncated_frac"] == pytest.approx(1 / 3, abs=1e-4)
    assert row["metrics"]["mrr@10"] == pytest.approx((1 + 1 / 3 + 1 / 2) / 3, abs=1e-4)


def test_untracked_files_do_not_make_runs_dirty(tmp_path, monkeypatch):
    import subprocess

    from vector.eval import run

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    monkeypatch.setattr(run, "REPO_ROOT", tmp_path)

    (tmp_path / "results.jsonl").write_text("{}\n")  # untracked
    assert run.git_info()["dirty"] is False
    (tmp_path / "code.py").write_text("x = 2\n")  # tracked change
    assert run.git_info()["dirty"] is True


def test_hnsw_index_spec_and_ann_recall(dataset, bow, tmp_path):
    flat = run_experiment(dataset, WholeDocumentChunker(), bow, k_chunks=10, cache_dir=tmp_path)
    hnsw = run_experiment(
        dataset, WholeDocumentChunker(), bow, k_chunks=10, cache_dir=tmp_path, index_spec="hnsw-4-16-16"
    )
    assert flat["index"] == "flat" and flat["ann_recall@10"] == pytest.approx(1.0)
    # Three documents: the beam covers the whole graph, so HNSW must match exact search.
    assert hnsw["index"] == "hnsw-4-16-16" and hnsw["ann_recall@10"] == pytest.approx(1.0)
    assert hnsw["metrics"] == flat["metrics"]


def test_parse_index():
    from vector.eval.run import parse_index
    from vector.index import FlatIndex, HNSWIndex

    assert isinstance(parse_index("flat", 8), FlatIndex)
    index = parse_index("hnsw-16-200-64", 8)
    assert isinstance(index, HNSWIndex) and (index.M, index.ef_construction, index.ef_search) == (16, 200, 64)
    assert type(parse_index("faiss-hnsw-8-100-32", 8)).__name__ == "FaissHNSWIndex"
    for bad in ("hnsw", "hnsw-16-200", "ivf-1-2-3", "hnsw-a-b-c", "faiss-16-200-64", "x-hnsw-1-2-3"):
        with pytest.raises(ValueError):
            parse_index(bad, 8)


def test_ef_search_sweep_reuses_the_graph(dataset, bow, tmp_path):
    from vector.eval.run import evaluate_index, prepare

    prepared = prepare(dataset, WholeDocumentChunker(), bow, tmp_path)
    built: dict = {}
    a = evaluate_index(dataset, WholeDocumentChunker(), bow, prepared, "hnsw-4-16-8", 10, built)
    b = evaluate_index(dataset, WholeDocumentChunker(), bow, prepared, "hnsw-4-16-32", 10, built)
    assert list(built) == ["hnsw-4-16"]  # one graph, two ef_search settings
    assert a["build_seconds"] == b["build_seconds"]
    assert built["hnsw-4-16"][0].ef_search == 32


def test_per_query_scores_average_to_the_reported_ndcg(dataset, bow, tmp_path):
    from vector.eval.run import evaluate_index, prepare

    prepared = prepare(dataset, WholeDocumentChunker(), bow, tmp_path)
    row = evaluate_index(dataset, WholeDocumentChunker(), bow, prepared, per_query=True)
    scores = row["per_query_ndcg@10"]
    assert set(scores) == {"q1", "q2", "q3"}
    assert sum(scores.values()) / 3 == pytest.approx(row["metrics"]["ndcg@10"], abs=1e-3)
    assert "per_query_ndcg@10" not in run_experiment(dataset, WholeDocumentChunker(), bow, cache_dir=tmp_path)
