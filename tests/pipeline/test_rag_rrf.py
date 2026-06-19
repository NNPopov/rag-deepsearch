"""QR4 — Reciprocal Rank Fusion (rag.search.rrf). Pure unit, no DB/network.

Basis: Rag_query_architecture.md §4. RRF in Python: score = Σ 1/(k + rank), k=60 (as in S14),
merges N vector + M bm25 ranked lists. Dedup by chunk_id (UNIQUE).
Key things the red test pins:
  • a chunk in several lists sums its contributions → ranks higher than a single one;
  • the result is sorted by descending RRF score, and score is set on RetrievedChunk;
  • dedup by chunk_id: one object per chunk, no duplicates;
  • rank is 1-based (top of a list = rank 1 = contribution 1/(k+1)).
"""
import pytest

from rag.contracts import RetrievedChunk
from rag.search import rrf


def _chunk(cid: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, content=f"c{cid}", document_id=1, section_id=cid, source="s"
    )


def test_rrf_dedups_by_chunk_id():
    a = [_chunk(1), _chunk(2)]
    b = [_chunk(2), _chunk(3)]
    merged = rrf([a, b])
    ids = [c.chunk_id for c in merged]
    assert sorted(ids) == [1, 2, 3]          # 2 not duplicated
    assert len(merged) == 3


def test_rrf_sums_contributions_for_shared_chunk():
    # chunk 2 is present in both lists → total contribution higher than the single ones
    a = [_chunk(1), _chunk(2)]
    b = [_chunk(2), _chunk(3)]
    merged = rrf([a, b])
    by_id = {c.chunk_id: c.score for c in merged}
    # rank c2: 2 in list a, 1 in list b → 1/(60+2) + 1/(60+1)
    assert by_id[2] == pytest.approx(1 / 62 + 1 / 61)
    # c1: only rank1 in a → 1/61 ; c3: only rank2 in b → 1/62
    assert by_id[1] == pytest.approx(1 / 61)
    assert by_id[3] == pytest.approx(1 / 62)
    assert by_id[2] > by_id[1] > by_id[3]


def test_rrf_sorted_descending_and_shared_wins():
    a = [_chunk(1), _chunk(2)]
    b = [_chunk(2), _chunk(3)]
    merged = rrf([a, b])
    scores = [c.score for c in merged]
    assert scores == sorted(scores, reverse=True)
    assert merged[0].chunk_id == 2           # the shared chunk goes to the top


def test_rrf_custom_k():
    a = [_chunk(1)]
    merged = rrf([a], k=10)
    assert merged[0].score == pytest.approx(1 / 11)   # rank1, k=10


def test_rrf_empty_input():
    assert rrf([]) == []
    assert rrf([[], []]) == []
