"""QR13 (part 3) — FastAPI REST + SSE adapter over the web-free `rag` core.

External-dependency strategy: PURE UNIT — a fake `DeepSearch` (run/stream), no DB and no
network. We verify exactly what the transport is responsible for (Rag_query_architecture.md §7):
  • `POST /query` → 200 + serialized `DeepSearchResult` (sync core via threadpool);
  • `--document-ids` → `filters={"document_ids":[...]}` reach the core (both in run and stream);
  • `GET /query/stream` → SSE: each `Event` → `event: <type>` + `data: <model_dump_json>`,
    in the same order `stream()` emits them (SSE and A2A are two renderings of the SAME stream);
  • `event_to_sse` maps type→event name and body→JSON;
  • composition root `build_serve_app` assembles the app via injection (load/build/deep_search).
The core (`rag`) knows nothing of the transport: the fake implements only `.run()`/`.stream()`.
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
    """Core fake: records calls, returns a predefined final/stream (no DB/network)."""

    def __init__(self) -> None:
        self.run_calls: list = []
        self.stream_calls: list = []

    def run(self, question: str, *, filters=None):
        self.run_calls.append((question, filters))
        return _result()

    def stream(self, question: str, *, filters=None):
        self.stream_calls.append((question, filters))
        yield from _events()


def _parse_sse(text: str) -> list[dict]:
    """Parse the SSE body into a list of {'event':..., 'data':...} (skip ': ' ping comments)."""
    out: list[dict] = []
    text = text.replace("\r\n", "\n")          # sse-starlette sends CRLF — normalize
    for block in text.strip().split("\n\n"):
        ev: dict = {}
        for line in block.splitlines():
            if line.startswith(":"):           # SSE comment (ping) — ignore
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


# --- event_to_sse --------------------------------------------------------------------

def test_event_to_sse_maps_type_and_json():
    ev = PlanReady(plan=_plan())
    sse = event_to_sse(ev)
    assert sse["event"] == "plan_ready"
    assert json.loads(sse["data"]) == json.loads(ev.model_dump_json())


# --- GET /query/stream (SSE) ---------------------------------------------------------

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

    app = build_serve_app(load=lambda: sentinel, build=fake_build)
    c = TestClient(app)
    r = c.post("/query", json={"question": "q"})
    assert r.status_code == 200
    assert built == [sentinel]                 # config read once and handed to the builder
    assert fake.run_calls == [("q", None)]


def test_build_serve_app_accepts_explicit_deep_search(fake):
    # deep_search passed directly → load/build are not called (no network/DB)
    def boom(*a, **k):  # pragma: no cover
        raise AssertionError("must not be called")

    app = build_serve_app(deep_search=fake, load=boom, build=boom)
    c = TestClient(app)
    assert c.post("/query", json={"question": "q"}).status_code == 200
