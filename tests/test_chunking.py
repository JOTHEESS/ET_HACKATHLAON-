"""Page-window chunking: short pages stay whole, long pages split with
overlap, and no text is ever lost at a window boundary."""
import pytest

pytest.importorskip("chromadb")

from ingest.vector_builder import OVERLAP_WORDS, WORDS_PER_CHUNK, _split_page


def test_short_page_is_one_chunk():
    text = "pump bearing vibration " * 20  # 60 words
    assert _split_page(text) == [text]


def test_long_page_splits_with_overlap():
    words = [f"w{i}" for i in range(1000)]
    chunks = _split_page(" ".join(words))
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= WORDS_PER_CHUNK
    # consecutive chunks share exactly the overlap
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-OVERLAP_WORDS:] == second[:OVERLAP_WORDS]


def test_no_words_lost():
    words = [f"w{i}" for i in range(1000)]
    chunks = _split_page(" ".join(words))
    covered = set()
    for chunk in chunks:
        covered.update(chunk.split())
    assert covered == set(words)


def test_boundary_exact_multiple():
    words = [f"w{i}" for i in range(WORDS_PER_CHUNK)]
    assert len(_split_page(" ".join(words))) == 1
