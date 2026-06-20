"""ADR-0002 L6 (red): FastAPI REST+SSE over an ASYNC core — async port of QR13 (test_serve_api.py).

L6 drops the thread-pool bridge: `serve.api` calls the async core DIRECTLY —
  • `POST /query` → `result = await deep_search.run(...)` (no `run_in_threadpool`);
  • `GET /query/stream` → `async for event in deep_search.stream(...)` (no `iterate_in_threadpool`).
The `Event → SSE` mapping (`event_to_sse`) and the wire format are unchanged.

Pure unit: an ASYNC fake `DeepSearch` (async `run`, async-generator `stream`), no DB/network.
`TestClient` drives the event loop synchronously, so the test functions stay sync. RED before L6:
`api.py` bridges through the thread pool, which cannot drive a coroutine `run` / an async-gen `stream`.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from rag.contracts import (
    AnswerDelta,
    DeepSearchResult,
    ExpandReady,
    Final,
    PlanReady,
    ReflectResult,
    SearchPlan,
    SearchRound,
)
from serve.api import create_app, event_to_sse
from serve.app import build_serve_app

ANSWER = "Failover promotes a follower [Designing Data-Intensive Applications › 6]."
CITATION = "Designing Data-Intensive Applications › 6"


def _plan() -> SearchPlan:
    return SearchPlan(sub_questions=["how is failover handled"],
                      vector_queries=["leader failover"], bm25_queries=["failover"])


def _result() -> DeepSearchResult:
    return DeepSearchResult(
        answer=ANSWER, citations=[CITATION], chunks=[], blocks=[],
        plan=_plan(), iterations=1, trace=[], cost=None,
    )


def _events() -> list:
    return [
        PlanReady(plan=_plan()),
        SearchRound(iteration=1, vector_queries=["leader failover"],
                    bm25_queries=["failover"], new_chunks=3, total_chunks=3),
        ReflectResult(iteration=1, is_sufficient=True, gaps=[]),
        ExpandReady(section_ids=[6], blocks=1),
        AnswerDelta(text="Failover "),
        AnswerDelta(text="promotes a follower."),
        Final(result=_result()),
    ]


class FakeDeepSearch:
    """Async core fake: `run` is a coroutine, `stream` is an async generator (no DB/network)."""

    def __init__(self) -> None:
        self.run_calls: list = []
        self.stream_calls: list = []

    async def run(self, question: str, *, filters=None):
        self.run_calls.append((question, filters))
        return _result()

    async def stream(self, question: str, *, filters=None):
        self.stream_calls.append((question, filters))
        for ev in _events():
            yield ev


def _parse_sse(text: str) -> list[dict]:
    out: list[dict] = []
    text = text.replace("\r\n", "\n")
    for block in text.strip().split("\n\n"):
        ev: dict = {}
        for line in block.splitlines():
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                ev["event"] = line[len("event:"):].strip()
            elif line.startswith("data:"):
                ev["data"] = ev.get("data", "") + line[len("data:"):].strip()
        if ev:
            out.append(ev)
    return out


@pytest.fixture
def fake() -> FakeDeepSearch:
    return FakeDeepSearch()


@pytest.fixture
def client(fake) -> TestClient:
    return TestClient(create_app(deep_search=fake))


# --- POST /query ---------------------------------------------------------------------

def test_post_query_returns_serialized_result(client):
    r = client.post("/query", json={"question": "How is leader failover handled?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == ANSWER
    assert body["citations"] == [CITATION]
    assert body["iterations"] == 1


def test_post_query_without_document_ids_passes_no_filters(client, fake):
    client.post("/query", json={"question": "q"})
    assert fake.run_calls == [("q", None)]


def test_post_query_with_document_ids_builds_filters(client, fake):
    client.post("/query", json={"question": "q", "document_ids": [1, 2]})
    assert fake.run_calls == [("q", {"document_ids": [1, 2]})]


# --- event_to_sse (unchanged mapping) ------------------------------------------------

def test_event_to_sse_maps_type_and_json():
    ev = PlanReady(plan=_plan())
    sse = event_to_sse(ev)
    assert sse["event"] == "plan_ready"
    assert json.loads(sse["data"]) == json.loads(ev.model_dump_json())


# --- GET /query/stream (SSE over an async generator) ---------------------------------

def test_stream_emits_events_in_order(client):
    r = client.get("/query/stream", params={"question": "How is failover handled?"})
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert [e["event"] for e in events] == [
        "plan_ready", "search_round", "reflect", "expand_ready",
        "answer_delta", "answer_delta", "final",
    ]


def test_stream_final_carries_full_result(client):
    r = client.get("/query/stream", params={"question": "q"})
    final = _parse_sse(r.text)[-1]
    assert final["event"] == "final"
    payload = json.loads(final["data"])
    assert payload["result"]["answer"] == ANSWER
    assert payload["result"]["citations"] == [CITATION]


def test_stream_passes_document_ids_filter(client, fake):
    client.get("/query/stream", params={"question": "q", "document_ids": [3, 4]})
    assert fake.stream_calls == [("q", {"document_ids": [3, 4]})]


# --- composition root (build_serve_app) ----------------------------------------------

def test_build_serve_app_uses_injected_deep_search(fake):
    sentinel = object()
    built: list = []

    def fake_build(settings, **kw):
        built.append(settings)
        return fake

    app = build_serve_app(load=lambda: sentinel, build=fake_build, with_a2a=False)
    c = TestClient(app)
    r = c.post("/query", json={"question": "q"})
    assert r.status_code == 200
    assert built == [sentinel]
    assert fake.run_calls == [("q", None)]


def test_build_serve_app_accepts_explicit_deep_search(fake):
    def boom(*a, **k):  # pragma: no cover
        raise AssertionError("must not be called")

    app = build_serve_app(deep_search=fake, load=boom, build=boom, with_a2a=False)
    c = TestClient(app)
    assert c.post("/query", json={"question": "q"}).status_code == 200
