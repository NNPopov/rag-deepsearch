"""QR7/QR8/QR9 — deep-search loop steps that go through the LLM gateway (rag.steps).

Grounding: Rag_query_architecture.md §1/§3/§5, Rag_implementation_steps.md (QR7–QR9).
All three steps are tested on a MOCK gateway (no network, CLAUDE.md §5):
  • PLAN   (QR7): question → SearchPlan via completion(task="plan"); empty question is rejected;
  • REFLECT(QR8): source-aware — coverage by document_id goes into the prompt; the CRAG threshold drops
    low-score chunks; gaps → new_*_queries; is_sufficient is parsed;
  • SYNTH  (QR9): the prompt carries ExpandedSection full_text (not chunk fragments); citations [source];
    tokens are streamed via completion_stream.

The LLM output is parsed via Model.model_validate_json — malformed JSON → ValidationError (§3).
RED until Q6/Q7/Q8: the rag.steps module does not exist yet.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.contracts import ExpandedSection, Reflection, RetrievedChunk, SearchPlan


class FakeGateway:
    """Mock gateway: completion returns predefined text per task; completion_stream
    yields the given tokens. Records calls (task, messages) — we check WHAT went into the prompt."""

    def __init__(self, *, responses: dict | None = None, stream_tokens: list[str] | None = None):
        self.responses = responses or {}
        self.stream_tokens = stream_tokens or []
        self.completion_calls: list[tuple[str, list[dict]]] = []
        self.stream_calls: list[tuple[str, list[dict]]] = []

    def completion(self, *, task: str, messages: list[dict], **kw) -> str:
        self.completion_calls.append((task, messages))
        return self.responses[task]

    def completion_stream(self, *, task: str, messages: list[dict], **kw):
        self.stream_calls.append((task, messages))
        for t in self.stream_tokens:
            yield t


def _prompt_text(messages: list[dict]) -> str:
    """The whole prompt text (system+user) as one string — to check what landed in it."""
    return "\n".join(m["content"] for m in messages)


# --- QR7: PLAN -----------------------------------------------------------------------

def test_plan_parses_searchplan_and_routes_task():
    from rag.steps import plan

    raw = SearchPlan(
        sub_questions=["what is a partition?", "how is rebalancing done?"],
        vector_queries=["how does partitioning split keys"],
        bm25_queries=["partitioning", "rebalancing"],
    ).model_dump_json()
    gw = FakeGateway(responses={"plan": raw})

    out = plan("How does partitioning work?", gateway=gw)

    assert isinstance(out, SearchPlan)
    assert out.sub_questions == ["what is a partition?", "how is rebalancing done?"]
    assert out.bm25_queries == ["partitioning", "rebalancing"]
    assert gw.completion_calls, "PLAN did not call the gateway"
    task, messages = gw.completion_calls[0]
    assert task == "plan"                                   # task routing in model_map
    assert "How does partitioning work?" in _prompt_text(messages)  # question went into the prompt


def test_plan_rejects_empty_question():
    from rag.steps import plan

    gw = FakeGateway(responses={"plan": "{}"})
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(ValueError):
            plan(bad, gateway=gw)
    assert gw.completion_calls == []  # we don't call the gateway on an empty question


def test_plan_rejects_invalid_llm_json():
    from rag.steps import plan

    gw = FakeGateway(responses={"plan": "not json at all"})
    with pytest.raises(ValidationError):
        plan("valid question", gateway=gw)


# --- QR8: REFLECT (source-aware) -----------------------------------------------------

def _chunk(cid, doc_id, content, source, score=1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, content=content, document_id=doc_id,
        section_id=cid, source=source, score=score,
    )


def _plan() -> SearchPlan:
    return SearchPlan(
        sub_questions=["what is replication?", "what is partitioning?"],
        vector_queries=["replication"], bm25_queries=["replication"],
    )


def test_reflect_prompt_is_source_aware_and_parses():
    from rag.steps import reflect

    chunks = [
        _chunk(1, 10, "leader-follower replication keeps copies", "Doc A › 1"),
        _chunk(2, 20, "partitioning splits keys by range", "Doc B › 1"),
    ]
    raw = Reflection(
        is_sufficient=False,
        answered=["what is replication?"],
        gaps=["what is partitioning?"],
        new_vector_queries=["how partitioning rebalances"],
        new_bm25_queries=["rebalancing"],
    ).model_dump_json()
    gw = FakeGateway(responses={"reflect": raw})

    out = reflect("Compare replication and partitioning", _plan(), chunks, gateway=gw)

    assert isinstance(out, Reflection)
    assert out.is_sufficient is False
    assert out.gaps == ["what is partitioning?"]
    assert out.new_vector_queries == ["how partitioning rebalances"]  # gaps → new_*_queries

    task, messages = gw.completion_calls[0]
    assert task == "reflect"
    prompt = _prompt_text(messages)
    # source-aware: both documents and both chunks are visible to the critic
    assert "Doc A" in prompt and "Doc B" in prompt
    assert "leader-follower replication" in prompt
    assert "partitioning splits keys" in prompt
    # the plan's sub-questions go into the prompt
    assert "what is replication?" in prompt and "what is partitioning?" in prompt


def test_reflect_relevance_threshold_drops_low_score_chunks():
    from rag.steps import reflect

    chunks = [
        _chunk(1, 10, "strong relevant evidence", "Doc A › 1", score=0.9),
        _chunk(2, 10, "weak noisy evidence", "Doc A › 2", score=0.01),
    ]
    raw = Reflection(is_sufficient=True).model_dump_json()
    gw = FakeGateway(responses={"reflect": raw})

    reflect("q", _plan(), chunks, gateway=gw, relevance_threshold=0.5)

    prompt = _prompt_text(gw.completion_calls[0][1])
    assert "strong relevant evidence" in prompt      # above threshold — kept
    assert "weak noisy evidence" not in prompt        # below the CRAG threshold — dropped


def test_reflect_with_no_evidence_still_calls_gateway():
    from rag.steps import reflect

    raw = Reflection(is_sufficient=False, gaps=["everything"]).model_dump_json()
    gw = FakeGateway(responses={"reflect": raw})

    out = reflect("q", _plan(), [], gateway=gw)
    assert out.is_sufficient is False
    assert gw.completion_calls  # the critic is called even on empty results (it honestly says "not enough")


# --- QR9: SYNTH (stream + citations over the blocks' full_text) ---------------------------------

def _block(sid, full_text, source) -> ExpandedSection:
    return ExpandedSection(section_id=sid, document_id=1, full_text=full_text, source=source)


def test_synth_stream_yields_tokens_and_routes_task():
    from rag.steps import synth_stream

    gw = FakeGateway(stream_tokens=["Repli", "cation ", "copies data."])
    blocks = [_block(1, "Replication keeps copies of data on multiple nodes.", "DDIA › 5")]

    out = list(synth_stream("What is replication?", blocks, gateway=gw))

    assert out == ["Repli", "cation ", "copies data."]   # tokens are streamed as-is
    task, messages = gw.stream_calls[0]
    assert task == "synth"


def test_synth_prompt_uses_full_text_and_source_for_citations():
    from rag.steps import synth_stream

    gw = FakeGateway(stream_tokens=["x"])
    blocks = [
        _block(1, "Replication keeps copies of data on multiple nodes.", "DDIA › 5"),
        _block(2, "Partitioning splits a dataset across nodes by key range.", "DDIA › 6"),
    ]
    list(synth_stream("Compare them", blocks, gateway=gw))

    prompt = _prompt_text(gw.stream_calls[0][1])
    # synthesis is over the chapters' full_text, not over fragments
    assert "Replication keeps copies of data on multiple nodes." in prompt
    assert "Partitioning splits a dataset across nodes by key range." in prompt
    # sources are available for [source] citations
    assert "DDIA › 5" in prompt and "DDIA › 6" in prompt
    assert "[source]" in prompt  # instruction to cite in the [source] format


def test_synth_joins_stream_into_string():
    from rag.steps import synth

    gw = FakeGateway(stream_tokens=["Hello", ", ", "world."])
    out = synth("q", [_block(1, "material", "S › 1")], gateway=gw)
    assert out == "Hello, world."
