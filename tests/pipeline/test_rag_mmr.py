"""QR5 — Maximal Marginal Relevance (rag.search.mmr). Pure unit, no DB/network.

Basis: Rag_query_architecture.md §4/§5 (problem B — the top is all from one place).
MMR reranks candidates before EXPAND: λ·relevance − (1−λ)·max cos(c, already selected).
Relevance = RRF score (RetrievedChunk.score). Pairwise cosine in Python over vector — no
new dependencies. Key things the red test pins:
  • the first is picked by maximum relevance;
  • at λ=0.5 a diverse candidate beats a near-duplicate of the top (diversity works);
  • λ=1.0 → pure relevance (order by score);
  • top_n caps the output; fewer candidates than top_n → all are returned (reranked);
  • vector=None is not penalized by similarity (cos=0).
"""
import pytest

from rag.contracts import RetrievedChunk
from rag.search import mmr


def _chunk(cid: int, score: float, vector):
    return RetrievedChunk(
        chunk_id=cid, content=f"c{cid}", document_id=1, section_id=cid,
        source="s", score=score, vector=vector,
    )


def test_mmr_first_pick_is_most_relevant():
    cands = [_chunk(1, 0.5, [1.0, 0.0]), _chunk(2, 0.9, [0.0, 1.0])]
    out = mmr(cands, lambda_=0.5, top_n=1)
    assert [c.chunk_id for c in out] == [2]      # highest score first


def test_mmr_diversity_beats_near_duplicate():
    c1 = _chunk(1, 1.0, [1.0, 0.0])      # top
    c2 = _chunk(2, 0.9, [1.0, 0.0])      # near-duplicate of c1
    c3 = _chunk(3, 0.5, [0.0, 1.0])      # diverse, lower by score
    out = mmr([c1, c2, c3], lambda_=0.5, top_n=2)
    assert [c.chunk_id for c in out] == [1, 3]   # after c1 take the diverse c3, not the dup c2


def test_mmr_lambda_one_is_pure_relevance():
    c1 = _chunk(1, 1.0, [1.0, 0.0])
    c2 = _chunk(2, 0.9, [1.0, 0.0])
    c3 = _chunk(3, 0.5, [0.0, 1.0])
    out = mmr([c1, c2, c3], lambda_=1.0, top_n=3)
    assert [c.chunk_id for c in out] == [1, 2, 3]  # pure order by score


def test_mmr_top_n_caps_output():
    cands = [_chunk(i, 1.0 / i, [1.0, 0.0]) for i in range(1, 6)]
    out = mmr(cands, lambda_=0.5, top_n=2)
    assert len(out) == 2


def test_mmr_returns_all_when_fewer_than_top_n():
    cands = [_chunk(1, 0.9, [1.0, 0.0]), _chunk(2, 0.5, [0.0, 1.0])]
    out = mmr(cands, lambda_=0.5, top_n=10)
    assert len(out) == 2
    assert {c.chunk_id for c in out} == {1, 2}


def test_mmr_none_vector_not_penalized():
    c1 = _chunk(1, 1.0, [1.0, 0.0])
    c2 = _chunk(2, 0.9, None)            # no vector → cos=0, diversity is not penalized
    out = mmr([c1, c2], lambda_=0.5, top_n=2)
    assert [c.chunk_id for c in out] == [1, 2]


def test_mmr_empty_input():
    assert mmr([], lambda_=0.5, top_n=5) == []
