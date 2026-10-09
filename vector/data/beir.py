"""Download and load BEIR datasets (corpus, queries, relevance judgments)."""

from __future__ import annotations

import csv
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "datasets"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}\n\n{self.text}" if self.title else self.text


@dataclass
class BeirDataset:
    corpus: dict[str, Document]
    queries: dict[str, str]
    # query_id -> {doc_id: relevance grade}; only grades > 0 are kept
    qrels: dict[str, dict[str, int]]


def download(name: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Fetch and unzip a BEIR dataset if it isn't already on disk."""
    target = data_dir / name
    if (target / "corpus.jsonl").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"{name}.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(BEIR_URL.format(name=name), zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    zip_path.unlink()
    return target


def load(path: Path, split: str = "test") -> BeirDataset:
    corpus: dict[str, Document] = {}
    with open(path / "corpus.jsonl") as f:
        for line in f:
            row = json.loads(line)
            corpus[row["_id"]] = Document(row["_id"], row.get("title", ""), row["text"])

    qrels: dict[str, dict[str, int]] = {}
    with open(path / "qrels" / f"{split}.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for query_id, doc_id, score in reader:
            if int(score) > 0:
                qrels.setdefault(query_id, {})[doc_id] = int(score)

    # Only keep queries that have judgments for this split.
    queries: dict[str, str] = {}
    with open(path / "queries.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["_id"] in qrels:
                queries[row["_id"]] = row["text"]

    return BeirDataset(corpus, queries, qrels)


def load_scifact(data_dir: Path = DEFAULT_DATA_DIR) -> BeirDataset:
    return load(download("scifact", data_dir))
