"""Content-addressed disk cache for embeddings.

The key hashes the model name and every input text, so any change that would
change the vectors (a different model, a chunker tweak, one edited character)
produces a new key. A stale cache can't be read by mistake.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

from vector.data.beir import DEFAULT_DATA_DIR
from vector.embed.base import Embedder

DEFAULT_CACHE_DIR = DEFAULT_DATA_DIR / "cache" / "embeddings"

log = logging.getLogger(__name__)


def cache_key(model_name: str, texts: list[str]) -> str:
    h = hashlib.sha256()
    # Length-prefix every field so ["ab", "c"] and ["a", "bc"] hash differently.
    for field in (model_name, *texts):
        data = field.encode("utf-8")
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def _cache_path(cache_dir: Path, model_name: str, key: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model_name)
    return cache_dir / safe_model / f"{key}.npy"


def cached_embed(
    embedder: Embedder, texts: list[str], cache_dir: Path = DEFAULT_CACHE_DIR
) -> np.ndarray:
    key = cache_key(embedder.name, texts)
    path = _cache_path(cache_dir, embedder.name, key)
    if path.exists():
        log.info("cache hit: %d texts, %s", len(texts), path.name[:12])
        return np.load(path)

    log.info("cache miss: embedding %d texts with %s", len(texts), embedder.name)
    start = time.perf_counter()
    vectors = embedder.embed(texts)
    seconds = time.perf_counter() - start
    if vectors.shape != (len(texts), embedder.dim):
        raise ValueError(f"expected shape {(len(texts), embedder.dim)}, got {vectors.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file, then rename, so a crash never leaves a half-written cache entry.
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.save(f, vectors.astype(np.float32))
    os.replace(tmp, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "model": embedder.name,
                "count": len(texts),
                "dim": embedder.dim,
                "embed_seconds": round(seconds, 3),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
    )
    return vectors
