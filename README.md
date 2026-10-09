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

### Phase 2: chunking strategy × chunk size × embedding model

Three chunkers (fixed word window, sentence-packing, recursive) at 100 / 150 / 200 words with 20% overlap, plus whole documents, on two local models of the same size (384 dimensions): **all-MiniLM-L6-v2** (256-token limit) and **bge-small-en-v1.5** (512-token limit, queries prefixed with its retrieval instruction). Exact search, 20 configurations, all from commit `49e953d`.

nDCG@10 (best per model in bold):

| Chunking | MiniLM | Truncated | bge-small | Truncated |
|---|---:|---:|---:|---:|
| whole document | 0.645 | 71.0% | 0.713 | 8.8% |
| fixed 100 / 20 | **0.674** | 0.2% | 0.716 | 0% |
| fixed 150 / 30 | 0.652 | 15.2% | 0.713 | 0% |
| fixed 200 / 40 | 0.648 | 49.2% | 0.712 | 0% |
| sentence 100 / 20 | 0.661 | 0.1% | 0.704 | 0% |
| sentence 150 / 30 | 0.653 | 8.2% | **0.726** | 0% |
| sentence 200 / 40 | 0.658 | 43.8% | 0.708 | 0% |
| recursive 100 / 20 | 0.663 | 0.1% | 0.706 | 0% |
| recursive 150 / 30 | 0.654 | 10.1% | 0.717 | 0% |
| recursive 200 / 40 | 0.651 | 44.7% | 0.708 | 0% |

- **The model matters far more than chunking.** bge-small on whole documents (0.713) beats every MiniLM configuration (best 0.674). Across all ten bge-small configurations, nDCG@10 spans only 0.704–0.726.
- **For MiniLM, chunking mostly fixes truncation.** All three 100-word configurations (≤0.2% truncated) beat every larger one. bge-small reads 512 tokens, so whole abstracts are rarely cut (8.8%) and chunking has little left to fix.
- **Recursive ≈ sentence on this data.** Every SciFact document's only paragraph break is between title and abstract, so the recursive splitter falls through to sentences.
- **Caveat:** scores are deterministic, but 300 queries is a small sample. A gap of about 0.01 nDCG can come from a handful of queries, so the ordering among bge-small chunkers isn't established without a significance test.

Full table: `python -m vector.eval.report results/phase2.jsonl`. Reproduce:

```bash
G="whole fixed-100-20 fixed-150-30 fixed-200-40 sentence-100-20 sentence-150-30 sentence-200-40 recursive-100-20 recursive-150-30 recursive-200-40"
.venv/bin/python -m vector.eval.run --model minilm    --chunker $G --out results/phase2.jsonl
.venv/bin/python -m vector.eval.run --model bge-small --chunker $G --out results/phase2.jsonl
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
- [x] Chunking × embedding-model sweep (local models; OpenAI pending)
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
