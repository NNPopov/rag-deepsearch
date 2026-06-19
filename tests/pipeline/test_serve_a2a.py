"""QR14 (part 3) — A2A adapter (a2a-sdk) over the same web-free `rag` core.

A2A and SSE are two renderings of the SAME `Event` stream (Rag_query_architecture.md §7). Here
we verify what the adapter is responsible for, as a pure unit (no network/DB, no full JSON-RPC stack):
  • `_drive` — the spec-significant mapping: core Event stream → A2A task lifecycle via
    TaskUpdater (submit → working(answer_delta) → artifact(answer) → complete);
  • `build_agent_card` — the agent card: streaming-capability + deep-search skill;
  • `add_a2a_routes` — route factory composing into the SAME FastAPI app: Agent Card served
    at `/.well-known/agent-card.json`, JSON-RPC route registered;
  • `build_serve_app(with_a2a=True)` — REST+SSE and A2A on one app.

⚠ SDK version is 1.1.0: card path `/.well-known/agent-card.json` (not `agent.json`), stream
method `message/stream` (not `tasks/sendSubscribe`) — SDK drift, isolated by the adapter (§7).
Full JSON-RPC e2e (`message/stream` via dispatcher) deferred — like the Q12 full-loop.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from rag.contracts import (
    AnswerDelta,
    DeepSearchResult,
    Final,
    PlanReady,
    SearchPlan,
)
from serve.a2a import add_a2a_routes, build_agent_card
from serve.a2a import _drive  # spec-significant mapping — tested directly
from serve.app import build_serve_app
from serve.api import create_app

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
    def run(self, question, *, filters=None):
        return _result()

    def stream(self, question, *, filters=None):
        yield from _events()


class FakeUpdater:
    """Records awaited TaskUpdater calls (the A2A task lifecycle order)."""

    def __init__(self) -> None:
        self.calls: list = []

    async def submit(self, message=None):
        self.calls.append(("submit",))

    async def start_work(self, message=None):
        self.calls.append(("start_work",))

    def new_agent_message(self, parts, metadata=None):
        return ("msg", parts)                     # echo — we'll check the part text

    async def update_status(self, state, message=None, **kw):
        self.calls.append(("update_status", state, message))

    async def add_artifact(self, parts, name=None, **kw):
        self.calls.append(("add_artifact", parts, name))

    async def complete(self, message=None):
        self.calls.append(("complete",))


async def _aiter(items):
    for x in items:
        yield x


# --- _drive: Event → A2A task lifecycle mapping --------------------------------------

def test_drive_maps_event_stream_to_task_lifecycle():
    from a2a.types import TaskState

    upd = FakeUpdater()
    # default threshold is large → two small deltas batch into ONE working flush at the end
    asyncio.run(_drive(upd, _aiter(_events())))

    kinds = [c[0] for c in upd.calls]
    # start_work → (batch of working deltas) → answer artifact → complete
    # (the Task/submitted object is seeded by execute BEFORE _drive — the framework requires a Task before any status-update)
    assert kinds == [
        "start_work", "update_status", "add_artifact", "complete",
    ]
    # the single working-status carries the concatenated deltas
    deltas = [c for c in upd.calls if c[0] == "update_status"]
    assert len(deltas) == 1
    assert deltas[0][1] == TaskState.TASK_STATE_WORKING
    assert deltas[0][2][1][0].text == "Failover promotes a follower."   # message=("msg",[Part]) → .text
    # final artifact = the full answer (from Final.result.answer, not from the deltas)
    artifact = next(c for c in upd.calls if c[0] == "add_artifact")
    assert artifact[1][0].text == ANSWER
    assert artifact[2] == "answer"


def test_drive_batches_answer_deltas_by_threshold():
    # many small deltas + a low threshold → FEWER flushes than deltas, but text preserved whole
    deltas = ["Fa", "il", "ov", "er", " p", "ro", "mo", "te", "s ", "fo", "ll", "ow", "er", "."]
    events = (
        [PlanReady(plan=_plan())]
        + [AnswerDelta(text=d) for d in deltas]
        + [Final(result=_result())]
    )
    upd = FakeUpdater()
    asyncio.run(_drive(upd, _aiter(events), delta_chars=10))

    flushes = [c for c in upd.calls if c[0] == "update_status"]
    assert 1 < len(flushes) < len(deltas)                 # batching: fewer frames than deltas
    assert "".join(f[2][1][0].text for f in flushes) == "".join(deltas)   # not a byte lost


# --- Agent Card ----------------------------------------------------------------------

def test_build_agent_card_advertises_streaming_and_skill():
    card = build_agent_card()
    assert card.name
    assert card.capabilities.streaming is True
    assert any(s.id == "deep-search" for s in card.skills)


# --- Composing routes into the SAME app -----------------------------------------------

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
    # JSON-RPC lives at rpc_url ("/") and does NOT break REST: /query still responds
    assert a2a_client.post("/query", json={"question": "q"}).status_code == 200
    paths = {getattr(r, "path", None) for r in a2a_client.app.routes}
    assert "/.well-known/agent-card.json" in paths


def test_build_serve_app_mounts_both_rest_and_a2a():
    app = build_serve_app(deep_search=FakeDeepSearch(), load=None, build=None, with_a2a=True)
    c = TestClient(app)
    assert c.post("/query", json={"question": "q"}).status_code == 200
    assert c.get("/.well-known/agent-card.json").status_code == 200
