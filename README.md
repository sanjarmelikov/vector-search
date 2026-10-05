# Anchor

A vector retrieval service built around a from-scratch HNSW index, benchmarked
against exact search and FAISS on the [BEIR SciFact](https://github.com/beir-cellar/beir) dataset.

> Work in progress. Results below will be filled in from real benchmark runs.

## Architecture

```
client ──► Anchor API (FastAPI)
              ├──► Throttle (C++ / Redis token-bucket rate limiter)
              ├──► embed query (sentence-transformers / OpenAI)
              ├──► vector index (from-scratch HNSW | FAISS | exact)
              ├──► re-rank (cross-encoder)
              └──► optional grounded answer (OpenAI)
```

## Roadmap

- [x] BEIR dataset loader
- [x] Chunkers: fixed window, sentence-packing, recursive
- [x] Exact (brute-force) index baseline
- [x] recall@k / MRR@k evaluation
- [ ] Embeddings: local sentence-transformers + OpenAI, cached to disk
- [ ] From-scratch HNSW index
- [ ] FAISS HNSW comparison
- [ ] Chunking × embedding-model sweep
- [ ] Re-ranking
- [ ] FastAPI service, persistence, incremental indexing
- [ ] Throttle integration and load testing

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

SciFact (~8 MB) downloads into `datasets/` on first use.
