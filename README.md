# vector-search

A vector retrieval service built around a from-scratch HNSW index, benchmarked
against exact search and FAISS on the [BEIR SciFact](https://github.com/beir-cellar/beir) dataset.

> Work in progress. Every number below comes from a recorded run in
> [`results/`](results/), tagged with the commit that produced it.

## Architecture

```
client ──► vector-search API (FastAPI)
              ├──► Throttle (C++ / Redis token-bucket rate limiter)
              ├──► embed query (sentence-transformers / OpenAI)
              ├──► vector index (from-scratch HNSW | FAISS | exact)
              ├──► re-rank (cross-encoder)
              └──► optional grounded answer (OpenAI)
```

## Results

### Phase 1: exact search on SciFact (300 test queries, 5,183 docs)

| Model | Chunking | Chunks | Truncated | R@1 | R@10 | MRR@10 | nDCG@10 | Search p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | whole document | 5,183 | 71.0% | 0.482 | 0.783 | 0.605 | 0.645 | 0.10 ms |
| all-MiniLM-L6-v2 | fixed 200 words, 40 overlap | 8,184 | 49.2% | 0.474 | 0.802 | 0.605 | 0.648 | 0.16 ms |
| all-MiniLM-L6-v2 | fixed 100 words, 20 overlap | 15,153 | 0.2% | **0.518** | **0.827** | **0.630** | **0.674** | 0.43 ms |

- **Sanity check:** the whole-document nDCG@10 of 0.645 matches the published BEIR result for this model on SciFact.
- **Truncation:** MiniLM reads at most 256 tokens and silently drops the rest. Scientific text averages about 1.6 tokens per word (measured: 337 tokens vs 215 words per abstract), so 71% of abstracts and even half of the 200-word chunks get cut off. Chunks short enough to fit raise nDCG@10 by 0.029.
- **Latency:** exact search time grows linearly with the number of chunks (3× the chunks, about 4× the p50). That cost is what HNSW is meant to avoid at scale.

Reproduce:

```bash
.venv/bin/pip install -e '.[dev,embed]'
.venv/bin/python -m vector.eval.run --model minilm --chunker whole fixed-200-40 fixed-100-20
```

## Roadmap

- [x] BEIR dataset loader
- [x] Chunkers: fixed window, sentence-packing, recursive
- [x] Exact (brute-force) index baseline
- [x] recall@k / MRR@k / nDCG@k evaluation
- [x] Embeddings: local sentence-transformers + OpenAI, cached to disk
- [x] Exact-search baseline on SciFact (MiniLM)
- [ ] From-scratch HNSW index
- [ ] FAISS HNSW comparison
- [ ] Chunking × embedding-model sweep
- [ ] Re-ranking
- [ ] FastAPI service, persistence, incremental indexing
- [ ] Throttle integration and load testing

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,embed]'
.venv/bin/pytest   # offline; no models or API calls
```

SciFact (~8 MB) downloads into `datasets/` on first use. Embeddings are cached in
`datasets/cache/embeddings/`, so reruns skip the model. OpenAI runs need
`OPENAI_API_KEY` in a `.env` file at the repo root.
