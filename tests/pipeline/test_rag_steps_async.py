"""ADR-0002 L3 (red): async PLAN/REFLECT/SYNTH — async port of QR7/QR8/QR9 (test_rag_steps.py).

L3 flips `rag.steps` to async over the gateway's async surface (L1):
  • plan/reflect → `await gateway.acompletion(task=...)`;
  • synth_stream → async generator over `gateway.acompletion_stream(...)` (`async for` → `yield`);
  • synth → async facade that joins the async token stream into a string.
The parsing contract is unchanged (Model.model_validate_json → ValidationError on garbage, §3).

MOCK async gateway, no network (CLAUDE.md §5). RED before L3: the steps are synchronous and call
the sync `completion`/`completion_stream`.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag.contracts import ExpandedSection, Reflection, RetrievedChunk, SearchPlan


class FakeGateway:
    """Async mock: acompletion → predefined text per task; acompletion_stream → an async gen of tokens.
    Records (task, messages) — we check WHAT went into the prompt."""

    def __init__(self, *, responses: dict | None = None, stream_tokens: list[str] | None = None):
        self.responses = responses or {}
        self.stream_tokens = stream_tokens or []
        self.completion_calls: list[tuple[str, list[dict]]] = []
        self.stream_calls: list[tuple[str, list[dict]]] = []

    async def acompletion(self, *, task: str, messages: list[dict], **kw) -> str:
        self.completion_calls.append((task, messages))
        return self.responses[task]

    async def acompletion_stream(self, *, task: str, messages: list[dict], **kw):
        self.stream_calls.append((task, messages))
        for t in self.stream_tokens:
            yield t


def _prompt_text(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages)


# --- QR7 (async): PLAN ---------------------------------------------------------------

async def test_plan_parses_searchplan_and_routes_task():
    from rag.steps import plan

    raw = SearchPlan(
        sub_questions=["what is a partition?", "how is rebalancing done?"],
        vector_queries=["how does partitioning split keys"],
        bm25_queries=["partitioning", "rebalancing"],
    ).model_dump_json()
    gw = FakeGateway(responses={"plan": raw})

    out = await plan("How does partitioning work?", gateway=gw)

    assert isinstance(out, SearchPlan)
    assert out.bm25_queries == ["partitioning", "rebalancing"]
    task, messages = gw.completion_calls[0]
    assert task == "plan"
    assert "How does partitioning work?" in _prompt_text(messages)


async def test_plan_rejects_empty_question():
    from rag.steps import plan

    gw = FakeGateway(responses={"plan": "{}"})
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(ValueError):
            await plan(bad, gateway=gw)
    assert gw.completion_calls == []  # the gateway is not called on an empty question


async def test_plan_rejects_invalid_llm_json():
    from rag.steps import plan

    gw = FakeGateway(responses={"plan": "not json at all"})
    with pytest.raises(ValidationError):
        await plan("valid question", gateway=gw)


# --- QR8 (async): REFLECT (source-aware) ---------------------------------------------

def _chunk(cid, doc_id, content, source, score=1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, content=content, document_id=doc_id,
        section_id=cid, source=source, score=score,
    )


def _plan_obj() -> SearchPlan:
    return SearchPlan(
        sub_questions=["what is replication?", "what is partitioning?"],
        vector_queries=["replication"], bm25_queries=["replication"],
    )


async def test_reflect_prompt_is_source_aware_and_parses():
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

    out = await reflect("Compare replication and partitioning", _plan_obj(), chunks, gateway=gw)

    assert isinstance(out, Reflection)
    assert out.is_sufficient is False
    assert out.new_vector_queries == ["how partitioning rebalances"]

    task, messages = gw.completion_calls[0]
    assert task == "reflect"
    prompt = _prompt_text(messages)
    assert "Doc A" in prompt and "Doc B" in prompt
    assert "leader-follower replication" in prompt
    assert "what is replication?" in prompt and "what is partitioning?" in prompt


async def test_reflect_relevance_threshold_drops_low_score_chunks():
    from rag.steps import reflect

    chunks = [
        _chunk(1, 10, "strong relevant evidence", "Doc A › 1", score=0.9),
        _chunk(2, 10, "weak noisy evidence", "Doc A › 2", score=0.01),
    ]
    gw = FakeGateway(responses={"reflect": Reflection(is_sufficient=True).model_dump_json()})

    await reflect("q", _plan_obj(), chunks, gateway=gw, relevance_threshold=0.5)

    prompt = _prompt_text(gw.completion_calls[0][1])
    assert "strong relevant evidence" in prompt
    assert "weak noisy evidence" not in prompt


async def test_reflect_with_no_evidence_still_calls_gateway():
    from rag.steps import reflect

    raw = Reflection(is_sufficient=False, gaps=["everything"]).model_dump_json()
    gw = FakeGateway(responses={"reflect": raw})

    out = await reflect("q", _plan_obj(), [], gateway=gw)
    assert out.is_sufficient is False
    assert gw.completion_calls


# --- QR9 (async): SYNTH (async stream + citations over the blocks' full_text) --------

def _block(sid, full_text, source) -> ExpandedSection:
    return ExpandedSection(section_id=sid, document_id=1, full_text=full_text, source=source)


async def test_synth_stream_yields_tokens_and_routes_task():
    from rag.steps import synth_stream

    gw = FakeGateway(stream_tokens=["Repli", "cation ", "copies data."])
    blocks = [_block(1, "Replication keeps copies of data on multiple nodes.", "DDIA › 5")]

    out = [t async for t in synth_stream("What is replication?", blocks, gateway=gw)]

    assert out == ["Repli", "cation ", "copies data."]
    task, messages = gw.stream_calls[0]
    assert task == "synth"


async def test_synth_prompt_uses_full_text_and_source_for_citations():
    from rag.steps import synth_stream

    gw = FakeGateway(stream_tokens=["x"])
    blocks = [
        _block(1, "Replication keeps copies of data on multiple nodes.", "DDIA › 5"),
        _block(2, "Partitioning splits a dataset across nodes by key range.", "DDIA › 6"),
    ]
    _ = [t async for t in synth_stream("Compare them", blocks, gateway=gw)]

    prompt = _prompt_text(gw.stream_calls[0][1])
    assert "Replication keeps copies of data on multiple nodes." in prompt
    assert "Partitioning splits a dataset across nodes by key range." in prompt
    assert "DDIA › 5" in prompt and "DDIA › 6" in prompt
    assert "[source]" in prompt


async def test_synth_joins_stream_into_string():
    from rag.steps import synth

    gw = FakeGateway(stream_tokens=["Hello", ", ", "world."])
    out = await synth("q", [_block(1, "material", "S › 1")], gateway=gw)
    assert out == "Hello, world."
