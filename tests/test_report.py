import json

from vector.eval.report import load_rows, markdown_table


def row(chunker, ndcg, commit="abc", dirty=False):
    return {
        "model": "org/model", "chunker": chunker, "k_chunks": 100, "query_prefix": "",
        "n_chunks": 10, "truncated_frac": 0.0, "search_p50_ms": 0.1,
        "commit": commit, "dirty": dirty,
        "metrics": {"recall@1": 0.1, "recall@10": 0.2, "mrr@10": 0.3, "ndcg@10": ndcg},
    }


def test_latest_run_of_a_config_wins(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [row("whole", 0.5), row("whole", 0.6, "def")]) + "\n")
    rows = load_rows(path)
    assert len(rows) == 1 and rows[0]["commit"] == "def"


def test_table_sorted_best_first_and_marks_dirty():
    table = markdown_table([row("a", 0.5), row("b", 0.7, dirty=True)])
    body = table.splitlines()[2:]
    assert "| b |" in body[0] and "abc*" in body[0]
    assert "| a |" in body[1]
