"""QR10 — deep-search loop orchestration (rag.loop.DeepSearch).

Basis: Rag_query_architecture.md §1/§3/§5, Rag_implementation_steps.md (QR10/Q9).
The core is web-free and synchronous; two entry points: stream() (Iterator[Event]) and run() (DeepSearchResult).
Loop: PLAN → SEARCH(hybrid RRF) → REFLECT(source-aware) → LOOP → EXPAND(small2big) → SYNTH.

Strategy (CLAUDE.md §5): pure unit on FAKE searchers + FAKE gateway + FAKE expander
(rrf/mmr are the real pure functions from rag.search). No DB and no network.
We check exit conditions (sufficient / no new queries / no new chunks / max_iterations),
accumulation with dedup by chunk_id, equivalence run() == the last Final from stream(), and trace.
RED before Q9: module rag.loop does not exist yet.
"""
from __future__ import annotations

import pytest

from rag.contracts import (
    AnswerDelta,
    ExpandedSection,
    ExpandReady,
    Final,
    PlanReady,
    Reflection,
    ReflectResult,
    RetrievedChunk,
    SearchPlan,
    SearchRound,
)


# --- fakes ----------------------------------------------------------------------------

def _chunk(cid: int, sec: int, source: str, vec: list[float]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, content=f"content-{cid}", document_id=1,
        section_id=sec, source=source, vector=vec,
    )


class FakeSearcher:
    """search(query) → copies of pre-set chunks keyed by query text (otherwise empty)."""

    def __init__(self, by_query: dict[str, list[RetrievedChunk]]):
        self.by_query = by_query
        self.calls: list[tuple[str, dict | None]] = []
        self.many_calls: list[tuple[list[str], dict | None]] = []

    def search(self, query: str, *, filters: dict | None = None) -> list[RetrievedChunk]:
        self.calls.append((query, filters))
        return [c.model_copy() for c in self.by_query.get(query, [])]

    def search_many(
        self, queries: list[str], *, filters: dict | None = None
    ) -> list[list[RetrievedChunk]]:
        """Batch entry: one call for all queries of a round (delegates to per-query search → .calls intact)."""
        self.many_calls.append((list(queries), filters))
        return [self.search(q, filters=filters) for q in queries]


class FakeExpander:
    """expand(chunks) → one ExpandedSection per unique section_id (small2big stub)."""

    def __init__(self):
        self.calls: list[list[RetrievedChunk]] = []

    def expand(self, chunks: list[RetrievedChunk]) -> list[ExpandedSection]:
        self.calls.append(list(chunks))
        blocks: dict[int, ExpandedSection] = {}
        for c in chunks:
            if c.section_id is None:
                continue
            b = blocks.get(c.section_id)
            if b is None:
                b = ExpandedSection(
                    section_id=c.section_id, document_id=c.document_id,
                    full_text=f"FULLTEXT[{c.section_id}]", source=c.source, from_chunks=[],
                )
                blocks[c.section_id] = b
            b.from_chunks.append(c.chunk_id)
        return list(blocks.values())


class FakeGateway:
    """plan→fixed JSON; reflect→a sequence of JSON (pop, sticks on the last);
    synth→streams the given tokens. Records calls for assertions."""

    def __init__(self, *, plan_json: str, reflect_json: list[str], synth_tokens: list[str]):
        self._plan = plan_json
        self._reflect = list(reflect_json)
        self._synth = synth_tokens
        self.completion_calls: list[tuple[str, list[dict]]] = []
        self.stream_calls: list[tuple[str, list[dict]]] = []

    def completion(self, *, task: str, messages: list[dict], **kw) -> str:
        self.completion_calls.append((task, messages))
        if task == "plan":
            return self._plan
        if task == "reflect":
            return self._reflect.pop(0) if len(self._reflect) > 1 else self._reflect[0]
        raise KeyError(task)

    def completion_stream(self, *, task: str, messages: list[dict], **kw):
        self.stream_calls.append((task, messages))
        for t in self._synth:
            yield t

    def reflect_calls(self) -> int:
        return sum(1 for task, _ in self.completion_calls if task == "reflect")


def _plan_json(vq: list[str], bq: list[str]) -> str:
    return SearchPlan(sub_questions=["s1"], vector_queries=vq, bm25_queries=bq).model_dump_json()


def _reflect_json(*, sufficient: bool, new_vq=None, new_bq=None, gaps=None) -> str:
    return Reflection(
        is_sufficient=sufficient,
        gaps=gaps or [],
        new_vector_queries=new_vq or [],
        new_bm25_queries=new_bq or [],
    ).model_dump_json()


A = _chunk(1, 10, "DDIA › A", [1.0, 0.0])
B = _chunk(2, 20, "DDIA › B", [0.0, 1.0])
C = _chunk(3, 30, "DDIA › C", [1.0, 1.0])


def _build(*, gateway, vector, bm25=None, expander=None, **kw):
    from rag.loop import DeepSearch

    return DeepSearch(
        vector_searcher=vector,
        bm25_searcher=bm25 or FakeSearcher({}),
        expander=expander or FakeExpander(),
        gateway=gateway,
        **kw,
    )


# --- exit conditions ------------------------------------------------------------------

def test_run_stops_when_reflection_sufficient():
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[_reflect_json(sufficient=True)],
        synth_tokens=["Answer cites ", "[DDIA › A]", " end."],
    )
    vector = FakeSearcher({"q1": [A, B]})
    ds = _build(gateway=gw, vector=vector)

    result = ds.run("how does it work?")

    assert result.iterations == 1                       # one round — the critic is satisfied
    assert result.answer == "Answer cites [DDIA › A] end."
    assert {b.section_id for b in result.blocks} == {10, 20}   # EXPAND over both sections
    assert result.plan.vector_queries == ["q1"]
    # citation extracted from the answer (only A was cited)
    assert result.citations == ["DDIA › A"]
    # chunks carry source and the RRF score (trace of sources/scores)
    assert all(c.source and c.score > 0 for c in result.chunks)
    assert gw.reflect_calls() == 1
    assert [q for q, _ in vector.calls] == ["q1"]


def test_loop_requeries_until_sufficient_and_dedups():
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[
            _reflect_json(sufficient=False, new_vq=["q2"], gaps=["more"]),
            _reflect_json(sufficient=True),
        ],
        synth_tokens=["ok"],
    )
    vector = FakeSearcher({"q1": [A, B], "q2": [B, C]})  # B repeats → dedup
    ds = _build(gateway=gw, vector=vector)

    result = ds.run("q")

    assert result.iterations == 2
    assert [q for q, _ in vector.calls] == ["q1", "q2"]    # requery with a new phrasing
    assert {c.chunk_id for c in result.chunks} == {1, 2, 3}  # A,B,C — B not duplicated
    assert {b.section_id for b in result.blocks} == {10, 20, 30}


def test_loop_stops_at_max_iterations():
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[                                   # the critic is NEVER satisfied
            _reflect_json(sufficient=False, new_vq=["q2"]),
            _reflect_json(sufficient=False, new_vq=["q3"]),
            _reflect_json(sufficient=False, new_vq=["q4"]),
        ],
        synth_tokens=["x"],
    )
    vector = FakeSearcher({"q1": [A], "q2": [B], "q3": [C]})  # each round brings something new
    ds = _build(gateway=gw, vector=vector, max_iterations=2)

    result = ds.run("q")

    assert result.iterations == 2                        # hit the ceiling, did not loop forever
    assert [q for q, _ in vector.calls] == ["q1", "q2"]
    assert result.answer == "x"                          # reached SYNTH anyway


def test_loop_stops_when_no_new_queries():
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[_reflect_json(sufficient=False, gaps=["x"])],  # there is a gap, but no queries
        synth_tokens=["x"],
    )
    vector = FakeSearcher({"q1": [A, B]})
    ds = _build(gateway=gw, vector=vector)

    result = ds.run("q")

    assert result.iterations == 1            # no requeries → corpus exhausted, we stop
    assert gw.reflect_calls() == 1


def test_loop_stops_when_requery_brings_no_new_chunks():
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[
            _reflect_json(sufficient=False, new_vq=["q2"]),
            _reflect_json(sufficient=True),    # should not be needed
        ],
        synth_tokens=["x"],
    )
    vector = FakeSearcher({"q1": [A, B], "q2": [A, B]})  # q2 brings nothing new
    ds = _build(gateway=gw, vector=vector)

    result = ds.run("q")

    assert result.iterations == 2            # the second search happened…
    assert gw.reflect_calls() == 1           # …but there is no reflection on it (0 new → stop before REFLECT)
    assert {c.chunk_id for c in result.chunks} == {1, 2}


# --- stream() ↔ run(), trace ----------------------------------------------------------

def _fresh_gateway():
    return FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[
            _reflect_json(sufficient=False, new_vq=["q2"], gaps=["g"]),
            _reflect_json(sufficient=True),
        ],
        synth_tokens=["Hello ", "[DDIA › A]"],
    )


def test_stream_emits_events_in_order():
    ds = _build(gateway=_fresh_gateway(), vector=FakeSearcher({"q1": [A, B], "q2": [B, C]}))

    events = list(ds.stream("q"))

    assert isinstance(events[0], PlanReady)
    assert isinstance(events[-1], Final)
    types = [type(e).__name__ for e in events]
    # plan → search/reflect rounds → expand → answer tokens → final
    assert types[0] == "PlanReady"
    assert types.count("SearchRound") == 2
    assert types.count("ReflectResult") == 2
    assert "ExpandReady" in types
    assert any(isinstance(e, AnswerDelta) for e in events)
    # first SearchRound: 2 new, 2 total; second: 1 new (B is a dup), 3 total
    rounds = [e for e in events if isinstance(e, SearchRound)]
    assert (rounds[0].new_chunks, rounds[0].total_chunks) == (2, 2)
    assert (rounds[1].new_chunks, rounds[1].total_chunks) == (1, 3)
    reflects = [e for e in events if isinstance(e, ReflectResult)]
    assert reflects[-1].is_sufficient is True


def test_run_result_equals_final_from_stream():
    ds_stream = _build(gateway=_fresh_gateway(), vector=FakeSearcher({"q1": [A, B], "q2": [B, C]}))
    finals = [e.result for e in ds_stream.stream("q") if isinstance(e, Final)]
    assert len(finals) == 1

    ds_run = _build(gateway=_fresh_gateway(), vector=FakeSearcher({"q1": [A, B], "q2": [B, C]}))
    result = ds_run.run("q")

    assert result.model_dump() == finals[0].model_dump()   # single source of truth — stream()


def test_trace_holds_full_event_stream():
    ds = _build(gateway=_fresh_gateway(), vector=FakeSearcher({"q1": [A, B], "q2": [B, C]}))
    result = ds.run("q")

    trace_types = {type(e).__name__ for e in result.trace}
    assert {"PlanReady", "SearchRound", "ReflectResult", "ExpandReady", "AnswerDelta"} <= trace_types
    assert not any(isinstance(e, Final) for e in result.trace)   # no self-recursion


# --- citation accuracy: robust label matching (a) ----------------------------------

def test_cited_sources_matches_canonical_label():
    """The model reproduced the label verbatim [title › path] → exact match."""
    from rag.loop import _cited_sources

    answer = "We rely on [Designing Data-Intensive Applications › 3] here."
    blocks = [
        "Designing Data-Intensive Applications › 3",
        "Acing the System Design Interview › 4",
    ]
    assert _cited_sources(answer, blocks) == ["Designing Data-Intensive Applications › 3"]


def test_cited_sources_robust_to_comma_and_deep_path():
    """flash bug: comma instead of › and a deep number (1.4.6, 4.8). We match book+chapter."""
    from rag.loop import _cited_sources

    answer = (
        "CQRS [Acing the System Design Interview, 1.4.6]. "
        "CDC [Designing Data-Intensive Applications, 12]. "
        "Cache [Acing the System Design Interview, 4.8]."
    )
    blocks = [
        "Acing the System Design Interview › 1",
        "Designing Data-Intensive Applications › 12",
        "Acing the System Design Interview › 4",
        "Designing Data-Intensive Applications › 8",   # not mentioned → must not be included
    ]
    assert _cited_sources(answer, blocks) == [
        "Acing the System Design Interview › 1",
        "Designing Data-Intensive Applications › 12",
        "Acing the System Design Interview › 4",
    ]


def test_cited_sources_ignores_chapter_mentions_outside_brackets():
    """"chapter 8" in prose (outside brackets) is not a citation; we count only what is in [...]."""
    from rag.loop import _cited_sources

    answer = "As discussed in chapter 8, indexes help. We cite [DDIA › 3]."
    blocks = ["DDIA › 3", "DDIA › 8"]
    assert _cited_sources(answer, blocks) == ["DDIA › 3"]


def test_cited_sources_empty_when_no_brackets():
    from rag.loop import _cited_sources

    assert _cited_sources("no citations here", ["DDIA › 3"]) == []


def test_run_citations_robust_to_label_format():
    """e2e via run(): both sections are in the blocks, but only chapter 3 is cited (comma format)."""
    a = _chunk(1, 10, "DDIA › 3", [1.0, 0.0])
    b = _chunk(2, 20, "DDIA › 8", [0.0, 1.0])
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[_reflect_json(sufficient=True)],
        synth_tokens=["See ", "[DDIA, 3]", " only."],   # comma, only chapter 3
    )
    ds = _build(gateway=gw, vector=FakeSearcher({"q1": [a, b]}))

    result = ds.run("q")

    assert result.citations == ["DDIA › 3"]   # not the "all blocks" fallback, but an exact citation


def test_loop_batches_vector_queries_in_one_call_per_round():
    """§6: a round's vector queries go out in ONE search_many, not one by one (a single batch embedding)."""
    gw = FakeGateway(
        plan_json=_plan_json(["q1", "q1b"], []),
        reflect_json=[_reflect_json(sufficient=True)],
        synth_tokens=["x"],
    )
    vector = FakeSearcher({"q1": [A], "q1b": [B]})
    ds = _build(gateway=gw, vector=vector)

    ds.run("q")

    assert vector.many_calls == [(["q1", "q1b"], None)]   # one batch per round, both queries together


def test_filters_forwarded_to_searchers():
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], ["k1"]),
        reflect_json=[_reflect_json(sufficient=True)],
        synth_tokens=["x"],
    )
    vector = FakeSearcher({"q1": [A]})
    bm25 = FakeSearcher({"k1": [B]})
    ds = _build(gateway=gw, vector=vector, bm25=bm25)

    ds.run("q", filters={"document_ids": [1]})

    assert vector.calls[0][1] == {"document_ids": [1]}
    assert bm25.calls[0][1] == {"document_ids": [1]}
