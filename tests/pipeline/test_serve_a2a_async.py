"""ADR-0002 L6 (red): A2A adapter over an ASYNC core — async port of QR14 (test_serve_a2a.py).

L6 drops the thread-pool bridge in `serve.a2a`: `execute` feeds `_drive` the core's NATIVE async
stream directly (`self._deep_search.stream(question)`) instead of `iterate_in_threadpool(...)`. The
`_drive` mapping itself is unchanged — it already consumes an async iterator — so here we exercise it
over a real async-generator core (the L6 shape) plus the card / route composition over an async core.

Pure unit (no DB/network/full JSON-RPC stack). The full `execute` JSON-RPC e2e stays deferred to the
live driver (`work/q14_a2a_e2e.py`, ported at L7), as in the original QR14. RED before L6: `/query`
(api.py) cannot drive the async fake through the thread pool.

asyncio_mode=auto → `_drive` tests are plain `async def` (no asyncio.run).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag.contracts import (
    AnswerDelta,
    DeepSearchResult,
    Final,
    PlanReady,
    SearchPlan,
)
from serve.a2a import _drive, add_a2a_routes, build_agent_card
from serve.api import create_app
from serve.app import build_serve_app

ANSWER = "Failover promotes a follower [DDIA › 6]."


def _plan() -> SearchPlan:
    return SearchPlan(sub_questions=["q"], vector_queries=["vq"], bm25_queries=["bq"])


def _result() -> DeepSearchResult:
    return DeepSearchResult(answer=ANSWER, citations=["DDIA › 6"], chunks=[], blocks=[],
                            plan=_plan(), iterations=1, trace=[], cost=None)


def _events() -> list:
    return [
        PlanReady(plan=_plan()),
        AnswerDelta(text="Failover "),
        AnswerDelta(text="promotes a follower."),
        Final(result=_result()),
    ]


class FakeDeepSearch:
    """Async core fake: coroutine `run` + async-generator `stream` (the L6 native-async shape)."""

    async def run(self, question, *, filters=None):
        return _result()

    async def stream(self, question, *, filters=None):
        for ev in _events():
            yield ev


class FakeUpdater:
    """Records awaited TaskUpdater calls (the A2A task lifecycle order)."""

    def __init__(self) -> None:
        self.calls: list = []

    async def submit(self, message=None):
        self.calls.append(("submit",))

    async def start_work(self, message=None):
        self.calls.append(("start_work",))

    def new_agent_message(self, parts, metadata=None):
        return ("msg", parts)

    async def update_status(self, state, message=None, **kw):
        self.calls.append(("update_status", state, message))

    async def add_artifact(self, parts, name=None, **kw):
        self.calls.append(("add_artifact", parts, name))

    async def complete(self, message=None):
        self.calls.append(("complete",))


# --- _drive over the core's NATIVE async stream (L6: no iterate_in_threadpool) --------

async def test_drive_maps_native_async_stream_to_task_lifecycle():
    from a2a.types import TaskState

    upd = FakeUpdater()
    # the core's async generator is consumed directly (this is exactly what execute now passes)
    await _drive(upd, FakeDeepSearch().stream("q"))

    kinds = [c[0] for c in upd.calls]
    assert kinds == ["start_work", "update_status", "add_artifact", "complete"]
    deltas = [c for c in upd.calls if c[0] == "update_status"]
    assert len(deltas) == 1
    assert deltas[0][1] == TaskState.TASK_STATE_WORKING
    assert deltas[0][2][1][0].text == "Failover promotes a follower."
    artifact = next(c for c in upd.calls if c[0] == "add_artifact")
    assert artifact[1][0].text == ANSWER
    assert artifact[2] == "answer"


async def test_drive_batches_answer_deltas_by_threshold():
    async def _aiter(items):
        for x in items:
            yield x

    deltas = ["Fa", "il", "ov", "er", " p", "ro", "mo", "te", "s ", "fo", "ll", "ow", "er", "."]
    events = (
        [PlanReady(plan=_plan())]
        + [AnswerDelta(text=d) for d in deltas]
        + [Final(result=_result())]
    )
    upd = FakeUpdater()
    await _drive(upd, _aiter(events), delta_chars=10)

    flushes = [c for c in upd.calls if c[0] == "update_status"]
    assert 1 < len(flushes) < len(deltas)
    assert "".join(f[2][1][0].text for f in flushes) == "".join(deltas)


# --- Agent Card (unchanged) ----------------------------------------------------------

def test_build_agent_card_advertises_streaming_and_skill():
    card = build_agent_card()
    assert card.name
    assert card.capabilities.streaming is True
    assert any(s.id == "deep-search" for s in card.skills)


# --- Composing routes over an async core ---------------------------------------------

@pytest.fixture
def a2a_client() -> TestClient:
    app = create_app(deep_search=FakeDeepSearch())
    add_a2a_routes(app, FakeDeepSearch())
    return TestClient(app)


def test_agent_card_served_at_well_known(a2a_client):
    r = a2a_client.get("/.well-known/agent-card.json")
    assert r.status_code == 200
    assert r.json()["name"]


def test_jsonrpc_route_registered_alongside_rest(a2a_client):
    assert a2a_client.post("/query", json={"question": "q"}).status_code == 200
    paths = {getattr(r, "path", None) for r in a2a_client.app.routes}
    assert "/.well-known/agent-card.json" in paths


def test_build_serve_app_mounts_both_rest_and_a2a():
    app = build_serve_app(deep_search=FakeDeepSearch(), load=None, build=None, with_a2a=True)
    c = TestClient(app)
    assert c.post("/query", json={"question": "q"}).status_code == 200
    assert c.get("/.well-known/agent-card.json").status_code == 200
