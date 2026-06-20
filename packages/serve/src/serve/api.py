"""FastAPI adapter (Q13): REST `POST /query` + SSE `GET /query/stream` over the `rag` core.

A thin adapter (Rag_query_architecture.md §7). The core is NATIVELY ASYNC (ADR-0002), so the edge
calls it directly — no thread-pool bridge —
  • `POST /query` → `result = await deep_search.run(...)`;
  • `GET /query/stream` → `async for event in deep_search.stream(...)`, each `Event` → an SSE frame
    via `EventSourceResponse` (sse-starlette).

SSE and A2A are two renderings of ONE `Event` stream: the `Event → data:` mapping lives in one place
(`event_to_sse`), reused by the A2A adapter (Q14). The core is injected (`deep_search`):
the adapter itself does not read config and does not touch the DB — that is the composition root's job (`serve/app.py`).

ADR-0002 supersedes 0001 (`async-edge-sync-core`) for this path: the thread-pool bridge
(`run_in_threadpool`/`iterate_in_threadpool`) is gone; streaming flows through native async generators
end-to-end.
"""
from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from rag.contracts import DeepSearchResult, Event


class QueryRequest(BaseModel):
    """`POST /query` body: question + (optional) narrowing search by a set of document_id."""

    question: str
    document_ids: list[int] | None = None


def _filters(document_ids: list[int] | None) -> dict | None:
    """document_ids → the core's filters dict (as in the CLI: empty → None)."""
    return {"document_ids": document_ids} if document_ids else None


def event_to_sse(event: Event) -> dict:
    """`Event` → ServerSentEvent fields: name = the `type` discriminator, body = its JSON.

    A single mapping point — reused by both SSE and A2A (§7: one stream, two renderings).
    """
    return {"event": event.type, "data": event.model_dump_json()}


def create_rest_routes(app: FastAPI, deep_search) -> FastAPI:
    """Mounts the REST+SSE routes on app over the injected core. Returns the same app."""

    @app.post("/query")
    async def query(req: QueryRequest):
        # native async core — await it directly (ADR-0002: no thread-pool bridge)
        result: DeepSearchResult | None = await deep_search.run(
            req.question, filters=_filters(req.document_ids)
        )
        if result is None:
            return JSONResponse(status_code=502, content={"detail": "no result from deep-search core"})
        return result

    @app.get("/query/stream")
    async def query_stream(
        question: str,
        document_ids: list[int] | None = Query(default=None),
    ):
        filters = _filters(document_ids)

        async def event_source():
            # the core's async generator is consumed directly, each Event → an SSE frame
            async for event in deep_search.stream(question, filters=filters):
                yield event_to_sse(event)

        return EventSourceResponse(event_source())

    return app


def create_app(*, deep_search) -> FastAPI:
    """Builds a FastAPI app with REST+SSE over the injected `deep_search` (without reading config)."""
    app = FastAPI(title="RAG deep-search", version="0.0.0")
    return create_rest_routes(app, deep_search)
