"""Turn a results JSONL file into a comparison table, best first.

    python -m vector.eval.report results/phase2.jsonl
    python -m vector.eval.report results/phase2.jsonl --model minilm --sort recall@10

If a configuration was run more than once, only its latest row counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLUMNS = ["recall@1", "recall@10", "mrr@10", "ndcg@10"]


def load_rows(path: Path) -> list[dict]:
    latest: dict[tuple, dict] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                row["model"], row["chunker"], row.get("index", "flat"),
                row["k_chunks"], row.get("query_prefix", ""),
            )
            latest[key] = row  # later lines replace earlier runs of the same config
    return list(latest.values())


def markdown_table(rows: list[dict], sort: str = "ndcg@10") -> str:
    rows = sorted(rows, key=lambda r: r["metrics"][sort], reverse=True)
    lines = [
        "| Model | Chunking | Index | Chunks | Truncated | R@1 | R@10 | MRR@10 | nDCG@10 | ANN@10 | p50 ms | Commit |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        m = r["metrics"]
        commit = (r.get("commit") or "?") + ("*" if r.get("dirty") else "")
        lines.append(
            f"| {r['model'].split('/')[-1]} | {r['chunker']} | {r.get('index', 'flat')} | {r['n_chunks']:,} | "
            f"{r['truncated_frac']:.1%} | {m['recall@1']:.3f} | {m['recall@10']:.3f} | "
            f"{m['mrr@10']:.3f} | {m['ndcg@10']:.3f} | {r.get('ann_recall@10', 1.0):.3f} | "
            f"{r['search_p50_ms']:.2f} | {commit} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path)
    parser.add_argument("--model", help="only rows whose model name contains this")
    parser.add_argument("--sort", choices=COLUMNS, default="ndcg@10")
    args = parser.parse_args(argv)

    rows = load_rows(args.path)
    if args.model:
        rows = [r for r in rows if args.model in r["model"]]
    print(markdown_table(rows, args.sort))
    if any(r.get("dirty") for r in rows):
        print("\n* = run with uncommitted changes; rerun after committing before publishing")


if __name__ == "__main__":
    main()
