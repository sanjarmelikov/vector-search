"""SIFT1M: the classic speed benchmark for nearest-neighbor search.

1,000,000 base vectors and 10,000 query vectors, 128 dimensions each
(image descriptors, not text). SciFact is too small to show HNSW's speed
advantage; SIFT1M is big enough.

We search by cosine similarity on normalized vectors, like the rest of the
project, so the dataset's own ground truth (Euclidean) doesn't apply. Exact
search (FlatIndex) computes the answer key instead.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from vector.data.beir import DEFAULT_DATA_DIR

SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"  # about 160 MB


def read_fvecs(path: Path, n: int | None = None) -> np.ndarray:
    """Read the .fvecs format: each row is an int32 dimension d, then d float32 values."""
    raw = np.memmap(path, dtype=np.int32, mode="r")  # read lazily; the file can be 500 MB
    dim = int(raw[0])
    rows = raw.reshape(-1, dim + 1)
    if n is not None:
        rows = rows[:n]
    if not np.all(rows[:, 0] == dim):
        raise ValueError(f"{path} is not a valid .fvecs file")
    return np.ascontiguousarray(rows[:, 1:]).view(np.float32)


def download(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Fetch and extract SIFT1M if it isn't already on disk."""
    target = data_dir / "sift"
    if (target / "sift_base.fvecs").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "sift.tar.gz"
    if not archive.exists():
        partial = archive.with_suffix(".tmp")
        urllib.request.urlretrieve(SIFT_URL, partial)
        partial.replace(archive)  # only a complete download gets the real name
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, filter="data")  # "data" blocks paths escaping data_dir
    archive.unlink()
    return target


def load_sift(
    n_base: int | None = None, n_queries: int | None = None, data_dir: Path = DEFAULT_DATA_DIR
) -> tuple[np.ndarray, np.ndarray]:
    """The first n_base base vectors and first n_queries query vectors (all if None)."""
    path = download(data_dir)
    return read_fvecs(path / "sift_base.fvecs", n_base), read_fvecs(path / "sift_query.fvecs", n_queries)
