"""QR2 — query-side contracts (rag.contracts). Pure unit, no DB/network.

Basis: Rag_query_architecture.md §3. Three groups of contracts by origin:
  • from the DB (trusted — we check the shape): RetrievedChunk, ExpandedSection;
  • from the LLM (MUST validate/reject malformed): SearchPlan, Reflection;
  • stream/result: Event (union, discriminated by "type"), DeepSearchResult.

Key things the red test pins:
  • RetrievedChunk carries chunk_id/section_id/document_id/vector (does not drop them, unlike a thin sketch);
  • malformed/incomplete LLM JSON → ValidationError (not "silent" garbage);
  • SearchPlan strips empties and requires a non-empty one (vector+bm25);
  • the Event union parses by the "type" field into the correct subtype.
"""
import pytest
from pydantic import TypeAdapter, ValidationError

from rag.contracts import (
    DeepSearchResult,
    DocumentCoverage,
    Event,
    ExpandedSection,
    Final,
    PlanReady,
    Reflection,
    RetrievedChunk,
    SearchPlan,
    SearchRound,
)


# --- 3.1 From the DB: shape, defaults, nothing lost ----------------------------------

def test_retrieved_chunk_carries_ids_and_vector():
    c = RetrievedChunk(
        chunk_id=42,
        content="text",
        document_id=7,
        section_id=3,
        source="DDIA › ch.5",
        vector=[0.1, 0.2, 0.3],
    )
    assert c.chunk_id == 42
    assert c.document_id == 7
    assert c.section_id == 3            # small2big key — NOT lost
    assert c.vector == [0.1, 0.2, 0.3]  # needed for MMR


def test_retrieved_chunk_defaults():
    c = RetrievedChunk(chunk_id=1, content="x", document_id=1, section_id=None, source="s")
    assert c.score == 0.0               # RRF will set it on merge
    assert c.vector is None             # could a BM25 hit miss the vector? — no, but the default exists
    assert c.metadata == {}             # page_start/end for citations
    assert c.section_id is None         # NULL is allowed


def test_expanded_section_shape():
    e = ExpandedSection(
        section_id=10, document_id=2, full_text="big chapter text", source="DDIA › ch.5"
    )
    assert e.full_text == "big chapter text"
    assert e.from_chunks == []          # provenance is empty by default


# --- 3.2 From the LLM: parsing + strict validation -----------------------------------

def test_search_plan_parses_clean_json():
    plan = SearchPlan.model_validate_json(
        '{"sub_questions": ["q1", "q2"],'
        ' "vector_queries": ["semantic phrasing"],'
        ' "bm25_queries": ["keywords"]}'
    )
    assert plan.sub_questions == ["q1", "q2"]
    assert plan.vector_queries == ["semantic phrasing"]


def test_search_plan_strips_and_drops_empty_queries():
    plan = SearchPlan(
        sub_questions=["q"],
        vector_queries=["  spaced  ", "", "   "],
        bm25_queries=["kw"],
    )
    assert plan.vector_queries == ["spaced"]   # trimmed + empties dropped


def test_search_plan_rejects_all_empty_queries():
    # there is a sub-question, but NOT a single real query — ValueError
    with pytest.raises(ValidationError):
        SearchPlan(sub_questions=["q"], vector_queries=["", "  "], bm25_queries=[])


def test_search_plan_requires_a_sub_question():
    with pytest.raises(ValidationError):
        SearchPlan(sub_questions=[], vector_queries=["v"], bm25_queries=[])


def test_reflection_parses_with_coverage():
    r = Reflection.model_validate_json(
        '{"is_sufficient": false,'
        ' "answered": ["q1"],'
        ' "gaps": ["q2 not found"],'
        ' "coverage_by_document": [{"document": "DDIA", "covered": ["q1"], "missing": ["q2"]}],'
        ' "new_vector_queries": ["retry phrasing"],'
        ' "new_bm25_queries": []}'
    )
    assert r.is_sufficient is False
    assert r.gaps == ["q2 not found"]
    assert isinstance(r.coverage_by_document[0], DocumentCoverage)
    assert r.coverage_by_document[0].missing == ["q2"]


def test_reflection_minimal_defaults():
    r = Reflection(is_sufficient=True)
    assert r.answered == [] and r.gaps == [] and r.coverage_by_document == []
    assert r.new_vector_queries == [] and r.new_bm25_queries == []


def test_reflection_rejects_garbage_json():
    # required is_sufficient is missing → ValidationError, not "silent" garbage
    with pytest.raises(ValidationError):
        Reflection.model_validate_json('{"gaps": ["x"]}')


# --- 3.3 Event union: discrimination by "type" ---------------------------------------

def test_event_union_discriminates_by_type():
    adapter = TypeAdapter(Event)
    ev = adapter.validate_python(
        {"type": "plan_ready",
         "plan": {"sub_questions": ["q"], "vector_queries": ["v"], "bm25_queries": []}}
    )
    assert isinstance(ev, PlanReady)
    assert ev.plan.sub_questions == ["q"]


def test_event_union_search_round():
    adapter = TypeAdapter(Event)
    ev = adapter.validate_python(
        {"type": "search_round", "iteration": 1,
         "vector_queries": ["v"], "bm25_queries": ["b"],
         "new_chunks": 5, "total_chunks": 5}
    )
    assert isinstance(ev, SearchRound)
    assert ev.iteration == 1 and ev.total_chunks == 5


def test_event_union_rejects_unknown_type():
    adapter = TypeAdapter(Event)
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "nonsense"})


# --- 3.3 DeepSearchResult: the final contract ----------------------------------------

def test_deep_search_result_shape():
    plan = SearchPlan(sub_questions=["q"], vector_queries=["v"], bm25_queries=[])
    res = DeepSearchResult(answer="the answer", plan=plan, iterations=2)
    assert res.answer == "the answer"
    assert res.iterations == 2
    assert res.citations == [] and res.chunks == [] and res.blocks == []
    assert res.cost is None


def test_final_event_wraps_result():
    plan = SearchPlan(sub_questions=["q"], vector_queries=["v"], bm25_queries=[])
    final = Final(result=DeepSearchResult(answer="a", plan=plan, iterations=1))
    assert final.type == "final"
    assert final.result.answer == "a"
