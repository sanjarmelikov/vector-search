"""Charts for the Phase 4 benchmark (results/phase4.jsonl → SVG files).

    python -m vector.eval.plot results/phase4.jsonl --out results/

1. recall_vs_latency.svg: each ef_search setting is one point; up and to the left is better.
2. latency_vs_size.svg: how query time grows with the number of vectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

# Reference palette (dataviz skill): categorical slots 1–2 for the two HNSW
# implementations; exact search is the baseline, drawn in neutral ink.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {
    "hnsw": ("Our HNSW (Python)", "#2a78d6"),
    "faiss-hnsw": ("FAISS HNSW (C++)", "#eb6834"),
    "flat": ("Exact (NumPy)", INK_2),
}


def kind(spec: str) -> str:
    return spec if spec == "flat" else spec.rsplit("-", 3)[0]


def ef(spec: str) -> int:
    return int(spec.rsplit("-", 1)[1])


def load(path: Path) -> list[dict]:
    latest = {}
    for line in open(path):
        if line.strip():
            row = json.loads(line)
            latest[(row["n"], row["index"])] = row  # rerun of a config replaces the old row
    return list(latest.values())


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=30)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plain_log_axis(axis) -> None:
    """Log scale with ticks at 1-2-5 steps, written as plain numbers (0.05, 10,000), not 5×10⁻²."""
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def legend_above(ax) -> None:
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=3, loc="lower left",
              bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=2.4)


def recall_vs_latency(rows: list[dict], n: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    at_n = [r for r in rows if r["n"] == n]
    for k in ("hnsw", "faiss-hnsw"):
        pts = sorted((r for r in at_n if kind(r["index"]) == k), key=lambda r: ef(r["index"]))
        if not pts:
            continue
        name, color = SERIES[k]
        x, y = [r["search_p50_ms"] for r in pts], [r["recall@10"] for r in pts]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name, zorder=3)
        for r in pts:  # label each point with its ef_search
            ax.annotate(f"ef={ef(r['index'])}", (r["search_p50_ms"], r["recall@10"]),
                        textcoords="offset points", xytext=(6, -12), fontsize=8, color=INK_2)
    flat = [r for r in at_n if r["index"] == "flat"]
    if flat:
        ax.axvline(flat[0]["search_p50_ms"], color=INK_2, linewidth=1.5, linestyle="--", zorder=2,
                   label="Exact (recall 1.0)")
    ax.set_xscale("log")
    plain_log_axis(ax.xaxis)
    style(ax, f"Recall vs query time, SIFT {n:,} vectors (k = 10)",
          "median query time, ms (log scale; left = faster)", "recall@10 vs exact search")
    legend_above(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def latency_vs_size(rows: list[dict], ef_search: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    for k in ("flat", "hnsw", "faiss-hnsw"):
        pts = sorted(
            (r for r in rows if kind(r["index"]) == k and (k == "flat" or ef(r["index"]) == ef_search)),
            key=lambda r: r["n"],
        )
        if not pts:
            continue
        name, color = SERIES[k]
        ax.plot([r["n"] for r in pts], [r["search_p50_ms"] for r in pts], color=color, linewidth=2,
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                linestyle="--" if k == "flat" else "-", label=name, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_axis(ax.xaxis)
    plain_log_axis(ax.yaxis)
    style(ax, f"Query time vs number of vectors (HNSW at ef_search = {ef_search})",
          "vectors in the index (log scale)", "median query time, ms (log scale)")
    legend_above(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--n", type=int, help="size for the recall plot (default: largest with our HNSW)")
    parser.add_argument("--ef", type=int, default=64, help="ef_search for the size plot")
    args = parser.parse_args(argv)

    rows = load(args.path)
    n = args.n or max(r["n"] for r in rows if kind(r["index"]) == "hnsw")
    args.out.mkdir(parents=True, exist_ok=True)
    recall_vs_latency(rows, n, args.out / "recall_vs_latency.svg")
    latency_vs_size(rows, args.ef, args.out / "latency_vs_size.svg")
    print(f"wrote {args.out}/recall_vs_latency.svg (n={n:,}) and latency_vs_size.svg")


if __name__ == "__main__":
    main()
