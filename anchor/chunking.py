"""Chunking strategies that split documents into retrievable passages.

Sizes are measured in whitespace-separated words. That keeps the chunkers
independent of any one embedding model's tokenizer; roughly 1 word ~ 1.3 tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from anchor.data.beir import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


class Chunker(Protocol):
    name: str

    def split(self, text: str) -> list[str]: ...


def chunk_documents(docs: list[Document], chunker: Chunker) -> list[Chunk]:
    chunks = []
    for doc in docs:
        for i, piece in enumerate(chunker.split(doc.full_text)):
            chunks.append(Chunk(f"{doc.doc_id}#{i}", doc.doc_id, piece))
    return chunks


def _check_sizes(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")


class WholeDocumentChunker:
    """No chunking: each document is one passage. The baseline for chunkers."""

    name = "whole"

    def split(self, text: str) -> list[str]:
        return [text.strip()] if text.strip() else []


class FixedSizeChunker:
    """Sliding window of `size` words, advancing by `size - overlap`."""

    def __init__(self, size: int = 200, overlap: int = 40):
        _check_sizes(size, overlap)
        self.size, self.overlap = size, overlap
        self.name = f"fixed-{size}-{overlap}"

    def split(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        step = self.size - self.overlap
        chunks = []
        for start in range(0, len(words), step):
            chunks.append(" ".join(words[start : start + self.size]))
            if start + self.size >= len(words):
                break
        return chunks


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _pack(units: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack text units into chunks of at most `size` words.

    Consecutive chunks share trailing units totalling at most `overlap` words,
    so overlap never cuts a unit in half. A single unit longer than `size`
    becomes its own chunk; callers split such units first if that matters.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        n = len(unit.split())
        if current and current_len + n > size:
            chunks.append(" ".join(current))
            # Carry trailing units forward as overlap.
            carried, carried_len = [], 0
            for prev in reversed(current):
                m = len(prev.split())
                if carried_len + m > overlap:
                    break
                carried.insert(0, prev)
                carried_len += m
            current, current_len = carried, carried_len
        current.append(unit)
        current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks


class SentenceChunker:
    """Packs whole sentences up to `size` words; never splits mid-sentence."""

    def __init__(self, size: int = 200, overlap: int = 40):
        _check_sizes(size, overlap)
        self.size, self.overlap = size, overlap
        self.name = f"sentence-{size}-{overlap}"

    def split(self, text: str) -> list[str]:
        return _pack(split_sentences(text), self.size, self.overlap)


class RecursiveChunker:
    """Splits on the coarsest separator that yields pieces <= `size` words.

    Tries paragraphs, then lines, then sentences, then falls back to a fixed
    word window, so structure is preserved wherever the text allows it.
    """

    def __init__(self, size: int = 200, overlap: int = 40):
        _check_sizes(size, overlap)
        self.size, self.overlap = size, overlap
        self.name = f"recursive-{size}-{overlap}"

    def _atoms(self, text: str, level: int) -> list[str]:
        if len(text.split()) <= self.size:
            return [text.strip()] if text.strip() else []
        if level == 0:
            parts = re.split(r"\n\s*\n", text)
        elif level == 1:
            parts = text.split("\n")
        elif level == 2:
            parts = split_sentences(text)
        else:
            return FixedSizeChunker(self.size, 0).split(text)
        if len(parts) == 1:
            return self._atoms(text, level + 1)
        atoms = []
        for part in parts:
            atoms.extend(self._atoms(part, level + 1))
        return atoms

    def split(self, text: str) -> list[str]:
        return _pack(self._atoms(text, 0), self.size, self.overlap)
