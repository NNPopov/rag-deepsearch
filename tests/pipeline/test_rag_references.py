"""QR-ref — down-ranking references-list sections (References), without removal.

Basis: the §"tag + down-rank" discussion. In the corpus (especially Acing) there are
references-list subsections ("4.13.1 Caching references", "17.12 References") — they compete with
substantive chunks and sometimes win by score (observed live: chunk_id=1017, score 0.16 — a
references list about caching). Decision: do NOT cut them (they can be useful as "give me sources"),
but penalize the RRF score by a tunable multiplier `references_penalty` (1.0 = disabled).

The red test pins:
  • the heading detector: References/Bibliography/Further Reading in the tail → True; content → False;
  • the penalty multiplies the score only for references chunks; leaves normal ones untouched;
  • penalty=1.0 — no-op (full backward compatibility);
  • a high raw score can still break through (the penalty is a down-rank, not a cutoff).
"""
import pytest

from rag.contracts import RetrievedChunk
from rag.search import is_references_heading, penalize_references


# --- heading detector ----------------------------------------------------------------

@pytest.mark.parametrize("heading", [
    "References",
    "17.12 References",
    "4.13.1 Caching references",     # the word in the tail of a real Acing heading
    "Bibliography",
    "Further Reading",
    "10.13  References",
])
def test_is_references_heading_true(heading):
    assert is_references_heading(heading) is True


@pytest.mark.parametrize("heading", [
    "6.1 Single-Leader Replication",
    "Reference architecture",        # "reference" at the start — NOT a references list
    "Partitioning",
    "",
    None,
])
def test_is_references_heading_false(heading):
    assert is_references_heading(heading) is False


# --- score penalty -------------------------------------------------------------------

def _chunk(cid: int, score: float, *, is_references: bool = False) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, content=f"c{cid}", document_id=1, section_id=cid,
        source="s", score=score, is_references=is_references,
    )


def test_penalize_lowers_only_references():
    chunks = [_chunk(1, 0.10), _chunk(2, 0.10, is_references=True)]
    penalize_references(chunks, 0.3)
    by_id = {c.chunk_id: c.score for c in chunks}
    assert by_id[1] == pytest.approx(0.10)        # content untouched
    assert by_id[2] == pytest.approx(0.03)        # references × 0.3


def test_penalty_one_is_noop():
    chunks = [_chunk(1, 0.10, is_references=True), _chunk(2, 0.20)]
    penalize_references(chunks, 1.0)
    assert chunks[0].score == pytest.approx(0.10)
    assert chunks[1].score == pytest.approx(0.20)


def test_high_raw_score_still_beats_content_after_penalty():
    # a references chunk with a strong raw score (the query is genuinely about literature) breaks through
    content = _chunk(1, 0.05)
    refs = _chunk(2, 0.30, is_references=True)
    penalize_references([content, refs], 0.3)
    assert refs.score == pytest.approx(0.09)
    assert refs.score > content.score             # 0.09 > 0.05 — not lost
