import pytest

from anchor.chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    chunk_documents,
    split_sentences,
)
from anchor.data.beir import Document


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_fixed_window_and_overlap():
    chunks = FixedSizeChunker(size=10, overlap=3).split(words(24))
    assert [len(c.split()) for c in chunks] == [10, 10, 10]
    # Each chunk starts 7 words after the previous one.
    assert chunks[1].split()[0] == "w7"
    assert chunks[1].split()[:3] == chunks[0].split()[-3:]
    assert chunks[-1].split()[-1] == "w23"


def test_fixed_no_trailing_duplicate_chunk():
    # 20 words, size 10, step 10: exactly two chunks, not a third empty/overlap-only one.
    assert len(FixedSizeChunker(size=10, overlap=0).split(words(20))) == 2


def test_fixed_short_and_empty_text():
    assert FixedSizeChunker(size=10, overlap=2).split("a b c") == ["a b c"]
    assert FixedSizeChunker().split("   ") == []


@pytest.mark.parametrize("cls", [FixedSizeChunker, SentenceChunker, RecursiveChunker])
def test_invalid_sizes(cls):
    with pytest.raises(ValueError):
        cls(size=10, overlap=10)
    with pytest.raises(ValueError):
        cls(size=0, overlap=0)


def test_split_sentences():
    text = "Cells divide. They grow (sometimes) fast! Is p < 0.05? Yes. e.g. this."
    assert split_sentences(text) == [
        "Cells divide.",
        "They grow (sometimes) fast!",
        "Is p < 0.05?",
        "Yes. e.g. this.",
    ]


def test_sentence_chunker_keeps_sentences_whole():
    sentences = [f"{words(6, prefix=f'S{i}_')}." for i in range(5)]
    chunks = SentenceChunker(size=14, overlap=7).split(" ".join(sentences))
    for chunk in chunks:
        assert len(chunk.split()) <= 14
        assert chunk.endswith(".")
    # Overlap carries the last whole sentence into the next chunk.
    assert chunks[0].split(". ")[-1].rstrip(".") == chunks[1].split(". ")[0]


def test_recursive_prefers_paragraph_boundaries():
    para_a, para_b = words(8, "a"), words(8, "b")
    chunks = RecursiveChunker(size=10, overlap=0).split(f"{para_a}\n\n{para_b}")
    assert chunks == [para_a, para_b]


def test_recursive_falls_back_to_word_window():
    chunks = RecursiveChunker(size=10, overlap=0).split(words(25))
    assert all(len(c.split()) <= 10 for c in chunks)
    assert " ".join(chunks).split() == words(25).split()


def test_chunk_documents_ids():
    docs = [Document("d1", "Title", words(15)), Document("d2", "", "short")]
    chunks = chunk_documents(docs, FixedSizeChunker(size=10, overlap=0))
    assert [c.chunk_id for c in chunks] == ["d1#0", "d1#1", "d2#0"]
    assert chunks[0].text.startswith("Title")
