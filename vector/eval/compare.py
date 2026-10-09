"""Is one system really better than another, or is the gap luck? Paired tests on nDCG@10.

    python -m vector.eval.run --model bge-small --chunker whole sentence-150-30 \\
        --per-query --out results/phase2_per_query.jsonl
    python -m vector.eval.compare results/phase2_per_query.jsonl --base bge-small:whole

Every system answers the same queries, so we compare them query by query
(a *paired* test). The randomization test asks: if the two systems were
equally good, then which one scored which on each query is a coin flip, so
how often would random flips produce a gap at least as large as the real one?
That fraction is the p-value. With many comparisons against one baseline,
some look "significant" by luck alone, so p-values are Holm-corrected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vector.eval.report import load_rows

FIELD = "per_query_ndcg@10"


def paired_randomization_test(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Mean of (b - a) and its two-sided p-value under random sign flips."""
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    observed = diffs.mean()
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_resamples, len(diffs)))
    null = (signs * diffs).mean(axis=1)
    # +1 on both sides counts the observed split itself, so p is never exactly 0.
    extreme = np.sum(np.abs(null) >= abs(observed) - 1e-12)
    return float(observed), float((extreme + 1) / (n_resamples + 1))


def bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Confidence interval for mean(b - a), resampling queries with replacement."""
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(diffs), size=(n_resamples, len(diffs)))
    means = diffs[idx].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def holm(p_values: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as given)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def label(row: dict) -> str:
    base = f"{row['model'].split('/')[-1]}:{row['chunker']}:{row.get('index', 'flat')}"
    if row.get("rerank"):
        base += f" + {row['rerank'].split('/')[-1]}@{row['rerank_depth']}"
    return base


def matches(row: dict, selector: str) -> bool:
    """'model[:chunker[:index[:rerank]]]', e.g. 'bge-small:whole' or 'bge-small:whole:flat:none'.

    model and rerank match as case-insensitive substrings; 'none' means not re-ranked.
    """
    parts = selector.split(":")
    if parts[0].lower() not in row["model"].lower():
        return False
    if len(parts) > 1 and parts[1] != row["chunker"]:
        return False
    if len(parts) > 2 and parts[2] != row.get("index", "flat"):
        return False
    if len(parts) > 3:
        rr = row.get("rerank")
        return rr is None if parts[3] == "none" else rr is not None and parts[3].lower() in rr.lower()
    return True


def compare(
    rows: list[dict], base_selector: str, against: str | None = None, alpha: float = 0.05
) -> str:
    """Test every row matching `against` (default: the baseline's model) against the baseline.

    Only the comparisons actually made count toward the Holm correction, so
    keep `against` to the question being asked.
    """
    rows = [r for r in rows if FIELD in r]
    bases = [r for r in rows if matches(r, base_selector)]
    if len(bases) != 1:
        raise ValueError(f"--base {base_selector!r} matched {len(bases)} rows with per-query scores; need 1")
    base = bases[0]
    queries = sorted(base[FIELD])
    a = np.array([base[FIELD][q] for q in queries])

    against = against or base_selector.split(":")[0]
    others = [r for r in rows if r is not base and matches(r, against) and sorted(r[FIELD]) == queries]
    tests = []
    for row in others:
        b = np.array([row[FIELD][q] for q in queries])
        diff, p = paired_randomization_test(a, b)
        tests.append((row, b.mean(), diff, bootstrap_ci(a, b), p))
    adjusted = holm([t[4] for t in tests])

    lines = [
        f"Baseline: {label(base)}  nDCG@10 = {a.mean():.3f}  ({len(queries)} queries)",
        "",
        "| System | nDCG@10 | Δ vs base | 95% CI of Δ | p | p (Holm) | Verdict |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for (row, mean, diff, (lo, hi), p), p_holm in sorted(zip(tests, adjusted), key=lambda t: -t[0][2]):
        verdict = ("better" if diff > 0 else "worse") if p_holm < alpha else "no clear difference"
        lines.append(
            f"| {label(row)} | {mean:.3f} | {diff:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
            f"{p:.3f} | {p_holm:.3f} | {verdict} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path)
    parser.add_argument("--base", required=True, help="baseline, e.g. 'bge-small:whole' or 'bge-small:whole:flat:none'")
    parser.add_argument("--against", help="which rows to test (selector); default: the baseline's model")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)
    print(compare(load_rows(args.path), args.base, args.against, args.alpha))


if __name__ == "__main__":
    main()
