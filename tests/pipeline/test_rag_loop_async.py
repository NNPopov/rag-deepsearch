"""ADR-0002 L4 (red): async DeepSearch loop — async port of QR10 (test_rag_loop.py).

L4 flips `rag.loop.DeepSearch` to async over the now-async steps/searchers (L2/L3):
  • stream() → an ASYNC generator (`async def` + `yield`), awaiting plan/reflect, the searchers
    (`await search_many`/`await search`), the expander (`await expand`) and `async for` over synth_stream;
  • run() → `async def` driving `async for ev in self.stream(...)`.
The pure helpers `_cited_sources`/rrf/mmr stay SYNC — their unit tests live in the (unchanged)
test_rag_loop.py and are not duplicated here.

Pure unit: FAKE async searchers + FAKE async gateway + FAKE async expander (rrf/mmr are the real
pure functions). No DB, no network. RED before L4: the loop is synchronous.
"""
from __future__ import annotations

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


# --- fakes (async) -------------------------------------------------------------------

def _chunk(cid: int, sec: int, source: str, vec: list[float]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, content=f"content-{cid}", document_id=1,
        section_id=sec, source=source, vector=vec,
    )


class FakeSearcher:
    """Async search/search_many → copies of pre-set chunks keyed by query text."""

    def __init__(self, by_query: dict[str, list[RetrievedChunk]]):
        self.by_query = by_query
        self.calls: list[tuple[str, dict | None]] = []
        self.many_calls: list[tuple[list[str], dict | None]] = []

    async def search(self, query: str, *, filters: dict | None = None) -> list[RetrievedChunk]:
        self.calls.append((query, filters))
        return [c.model_copy() for c in self.by_query.get(query, [])]

    async def search_many(
        self, queries: list[str], *, filters: dict | None = None
    ) -> list[list[RetrievedChunk]]:
        self.many_calls.append((list(queries), filters))
        return [await self.search(q, filters=filters) for q in queries]


class FakeExpander:
    """Async expand(chunks) → one ExpandedSection per unique section_id (small2big stub)."""

    def __init__(self):
        self.calls: list[list[RetrievedChunk]] = []

    async def expand(self, chunks: list[RetrievedChunk]) -> list[ExpandedSection]:
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
    """Async: acompletion (plan/reflect) + acompletion_stream (synth as an async gen)."""

    def __init__(self, *, plan_json: str, reflect_json: list[str], synth_tokens: list[str]):
        self._plan = plan_json
        self._reflect = list(reflect_json)
        self._synth = synth_tokens
        self.completion_calls: list[tuple[str, list[dict]]] = []
        self.stream_calls: list[tuple[str, list[dict]]] = []

    async def acompletion(self, *, task: str, messages: list[dict], **kw) -> str:
        self.completion_calls.append((task, messages))
        if task == "plan":
            return self._plan
        if task == "reflect":
            return self._reflect.pop(0) if len(self._reflect) > 1 else self._reflect[0]
        raise KeyError(task)

    async def acompletion_stream(self, *, task: str, messages: list[dict], **kw):
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


async def _collect(stream):
    return [ev async for ev in stream]


# --- exit conditions -----------------------------------------------------------------

async def test_run_stops_when_reflection_sufficient():
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[_reflect_json(sufficient=True)],
        synth_tokens=["Answer cites ", "[DDIA › A]", " end."],
    )
    vector = FakeSearcher({"q1": [A, B]})
    ds = _build(gateway=gw, vector=vector)

    result = await ds.run("how does it work?")

    assert result.iterations == 1
    assert result.answer == "Answer cites [DDIA › A] end."
    assert {b.section_id for b in result.blocks} == {10, 20}
    assert result.citations == ["DDIA › A"]
    assert all(c.source and c.score > 0 for c in result.chunks)
    assert gw.reflect_calls() == 1
    assert [q for q, _ in vector.calls] == ["q1"]


async def test_loop_requeries_until_sufficient_and_dedups():
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[
            _reflect_json(sufficient=False, new_vq=["q2"], gaps=["more"]),
            _reflect_json(sufficient=True),
        ],
        synth_tokens=["ok"],
    )
    vector = FakeSearcher({"q1": [A, B], "q2": [B, C]})
    ds = _build(gateway=gw, vector=vector)

    result = await ds.run("q")

    assert result.iterations == 2
    assert [q for q, _ in vector.calls] == ["q1", "q2"]
    assert {c.chunk_id for c in result.chunks} == {1, 2, 3}
    assert {b.section_id for b in result.blocks} == {10, 20, 30}


async def test_loop_stops_at_max_iterations():
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[
            _reflect_json(sufficient=False, new_vq=["q2"]),
            _reflect_json(sufficient=False, new_vq=["q3"]),
            _reflect_json(sufficient=False, new_vq=["q4"]),
        ],
        synth_tokens=["x"],
    )
    vector = FakeSearcher({"q1": [A], "q2": [B], "q3": [C]})
    ds = _build(gateway=gw, vector=vector, max_iterations=2)

    result = await ds.run("q")

    assert result.iterations == 2
    assert [q for q, _ in vector.calls] == ["q1", "q2"]
    assert result.answer == "x"


async def test_loop_stops_when_no_new_queries():
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[_reflect_json(sufficient=False, gaps=["x"])],
        synth_tokens=["x"],
    )
    ds = _build(gateway=gw, vector=FakeSearcher({"q1": [A, B]}))

    result = await ds.run("q")

    assert result.iterations == 1
    assert gw.reflect_calls() == 1


async def test_loop_stops_when_requery_brings_no_new_chunks():
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[
            _reflect_json(sufficient=False, new_vq=["q2"]),
            _reflect_json(sufficient=True),
        ],
        synth_tokens=["x"],
    )
    vector = FakeSearcher({"q1": [A, B], "q2": [A, B]})
    ds = _build(gateway=gw, vector=vector)

    result = await ds.run("q")

    assert result.iterations == 2
    assert gw.reflect_calls() == 1
    assert {c.chunk_id for c in result.chunks} == {1, 2}


# --- stream() ↔ run(), trace ---------------------------------------------------------

def _fresh_gateway():
    return FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[
            _reflect_json(sufficient=False, new_vq=["q2"], gaps=["g"]),
            _reflect_json(sufficient=True),
        ],
        synth_tokens=["Hello ", "[DDIA › A]"],
    )


async def test_stream_emits_events_in_order():
    ds = _build(gateway=_fresh_gateway(), vector=FakeSearcher({"q1": [A, B], "q2": [B, C]}))

    events = await _collect(ds.stream("q"))

    assert isinstance(events[0], PlanReady)
    assert isinstance(events[-1], Final)
    types = [type(e).__name__ for e in events]
    assert types[0] == "PlanReady"
    assert types.count("SearchRound") == 2
    assert types.count("ReflectResult") == 2
    assert "ExpandReady" in types
    assert any(isinstance(e, AnswerDelta) for e in events)
    rounds = [e for e in events if isinstance(e, SearchRound)]
    assert (rounds[0].new_chunks, rounds[0].total_chunks) == (2, 2)
    assert (rounds[1].new_chunks, rounds[1].total_chunks) == (1, 3)
    reflects = [e for e in events if isinstance(e, ReflectResult)]
    assert reflects[-1].is_sufficient is True


async def test_run_result_equals_final_from_stream():
    ds_stream = _build(gateway=_fresh_gateway(), vector=FakeSearcher({"q1": [A, B], "q2": [B, C]}))
    finals = [e.result for e in await _collect(ds_stream.stream("q")) if isinstance(e, Final)]
    assert len(finals) == 1

    ds_run = _build(gateway=_fresh_gateway(), vector=FakeSearcher({"q1": [A, B], "q2": [B, C]}))
    result = await ds_run.run("q")

    assert result.model_dump() == finals[0].model_dump()


async def test_trace_holds_full_event_stream():
    ds = _build(gateway=_fresh_gateway(), vector=FakeSearcher({"q1": [A, B], "q2": [B, C]}))
    result = await ds.run("q")

    trace_types = {type(e).__name__ for e in result.trace}
    assert {"PlanReady", "SearchRound", "ReflectResult", "ExpandReady", "AnswerDelta"} <= trace_types
    assert not any(isinstance(e, Final) for e in result.trace)


# --- citation accuracy via run() (the helper itself stays sync, tested in test_rag_loop.py) ---

async def test_run_citations_robust_to_label_format():
    a = _chunk(1, 10, "DDIA › 3", [1.0, 0.0])
    b = _chunk(2, 20, "DDIA › 8", [0.0, 1.0])
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], []),
        reflect_json=[_reflect_json(sufficient=True)],
        synth_tokens=["See ", "[DDIA, 3]", " only."],
    )
    ds = _build(gateway=gw, vector=FakeSearcher({"q1": [a, b]}))

    result = await ds.run("q")

    assert result.citations == ["DDIA › 3"]


async def test_loop_batches_vector_queries_in_one_call_per_round():
    gw = FakeGateway(
        plan_json=_plan_json(["q1", "q1b"], []),
        reflect_json=[_reflect_json(sufficient=True)],
        synth_tokens=["x"],
    )
    vector = FakeSearcher({"q1": [A], "q1b": [B]})
    ds = _build(gateway=gw, vector=vector)

    await ds.run("q")

    assert vector.many_calls == [(["q1", "q1b"], None)]


async def test_filters_forwarded_to_searchers():
    gw = FakeGateway(
        plan_json=_plan_json(["q1"], ["k1"]),
        reflect_json=[_reflect_json(sufficient=True)],
        synth_tokens=["x"],
    )
    vector = FakeSearcher({"q1": [A]})
    bm25 = FakeSearcher({"k1": [B]})
    ds = _build(gateway=gw, vector=vector, bm25=bm25)

    await ds.run("q", filters={"document_ids": [1]})

    assert vector.calls[0][1] == {"document_ids": [1]}
    assert bm25.calls[0][1] == {"document_ids": [1]}
