import numpy as np
import pytest

from vector.eval.compare import bootstrap_ci, compare, holm, paired_randomization_test


def test_identical_systems_are_not_different():
    a = np.random.default_rng(0).random(200)
    diff, p = paired_randomization_test(a, a.copy())
    assert diff == 0 and p == pytest.approx(1.0)


def test_consistent_small_gain_is_significant():
    # B beats A by 0.02 on every query: tiny, but perfectly consistent.
    a = np.random.default_rng(1).random(300)
    diff, p = paired_randomization_test(a, a + 0.02)
    assert diff == pytest.approx(0.02) and p < 0.001
    lo, hi = bootstrap_ci(a, a + 0.02)
    assert lo == pytest.approx(0.02) and hi == pytest.approx(0.02)


def test_large_gain_on_one_query_is_not_significant():
    # The same average gain, but all from a single query: that's luck, not a better system.
    a = np.zeros(300)
    b = a.copy()
    b[0] = 6.0  # mean gain 0.02
    diff, p = paired_randomization_test(a, b)
    assert diff == pytest.approx(0.02) and p > 0.5


def test_holm_by_hand():
    # Sorted p: 0.01 (x3 = 0.03), 0.02 (x2 = 0.04), 0.04 (x1 = 0.04), back in the input order.
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def row(chunker, scores):
    return {
        "model": "org/bge-small", "chunker": chunker, "index": "flat",
        "per_query_ndcg@10": {f"q{i}": s for i, s in enumerate(scores)},
    }


def test_compare_table_and_selector_errors():
    rng = np.random.default_rng(2)
    base = rng.random(100)
    rows = [row("whole", base), row("better", base + 0.05), row("same", base)]
    table = compare(rows, "bge-small:whole")
    assert "bge-small:better:flat" in table and "| better |" in table
    assert "no clear difference" in table.splitlines()[-1]  # 'same' sorts last
    with pytest.raises(ValueError):
        compare(rows, "bge-small")  # matches all three rows
