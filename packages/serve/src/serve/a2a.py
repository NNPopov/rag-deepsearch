"""A2A adapter (Q14): Agent Card + JSON-RPC `message/stream` over the SAME `rag` core.

A2A and SSE are two renderings of ONE `Event` stream (Rag_query_architecture.md §7): SSE maps
`Event → data:`, A2A maps the same stream into a task lifecycle (statuses + artifact). All of
the SDK specifics (`a2a-sdk`) are locked in here — the `rag` core knows nothing about them (§7, fork 2).

Assembly is via the route factory `add_a2a_routes_to_fastapi`: the Agent Card + JSON-RPC routes
are mounted on the SAME FastAPI app as REST+SSE (one app, as in the design).

⚠ SDK drift (version pinned to 1.1.0): the card path is `/.well-known/agent-card.json`
(not `agent.json`), the streaming method is `message/stream` (not `tasks/sendSubscribe`). This is exactly
the risk for the isolation of which the SDK is locked inside the adapter (§7).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.concurrency import iterate_in_threadpool

from a2a.helpers.proto_helpers import new_task
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Part,
    TaskState,
)
from a2a.utils import DEFAULT_RPC_URL, TransportProtocol

ANSWER_ARTIFACT = "answer"
DELTA_FLUSH_CHARS = 200          # A2A working-delta batching threshold: token→frame was chatty (459 frames)


def build_agent_card(*, url: str = "http://localhost:8000/") -> AgentCard:
    """Agent card (declares streaming + the deep-search skill). Serialized into Agent Card JSON."""
    return AgentCard(
        name="RAG deep-search",
        description="Answers questions over a corpus of books (PLAN-SEARCH-REFLECT-EXPAND-SYNTH) "
        "with citations to the sources.",
        version="0.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="deep-search",
                name="Deep search over the corpus",
                description="Hybrid search + iterative reflection → a grounded answer with citations.",
                tags=["rag", "search", "qa"],
            )
        ],
        supported_interfaces=[
            AgentInterface(url=url, protocol_binding=TransportProtocol.JSONRPC)
        ],
    )


async def _drive(updater, aevents, *, delta_chars: int = DELTA_FLUSH_CHARS) -> str:
    """Maps the core's Event stream → the A2A task lifecycle (the adapter's meaningful logic).

    start_work → `answer_delta`s accumulate into a buffer and are flushed as a working status in
    CHUNKS of `delta_chars` characters (the remainder — before the artifact): otherwise token=frame
    and the A2A stream is chatty. The final answer is shaped as the `answer` artifact → complete.
    `aevents` is an ASYNCHRONOUS iterator (in prod it is the sync `stream()` wrapped in
    `iterate_in_threadpool`: async edge, sync core). The Task object itself (submitted) is seeded by
    `execute` BEFORE the call — the framework requires a Task before any status-update.
    """
    await updater.start_work()

    async def _flush(text: str) -> None:
        await updater.update_status(
            TaskState.TASK_STATE_WORKING,
            message=updater.new_agent_message([Part(text=text)]),
        )

    answer = ""
    buffer = ""
    async for ev in aevents:
        if ev.type == "answer_delta":
            buffer += ev.text
            if len(buffer) >= delta_chars:      # threshold reached → one frame per chunk
                await _flush(buffer)
                buffer = ""
        elif ev.type == "final":
            answer = ev.result.answer
    if buffer:                                  # tail shorter than the threshold — send it
        await _flush(buffer)

    await updater.add_artifact([Part(text=answer)], name=ANSWER_ARTIFACT)
    await updater.complete()
    return answer


class RagAgentExecutor(AgentExecutor):
    """A2A executor: runs the web-free `rag` core and maps its `Event` stream into an A2A task."""

    def __init__(self, deep_search) -> None:
        self._deep_search = deep_search

    async def execute(self, context, event_queue) -> None:
        question = context.get_user_input()
        task_id, context_id = context.task_id, context.context_id
        # the framework requires a Task object in the queue BEFORE any status-update — we seed it (= submit)
        if context.current_task is None:
            await event_queue.enqueue_event(
                new_task(task_id, context_id, TaskState.TASK_STATE_SUBMITTED)
            )
        updater = TaskUpdater(event_queue, task_id, context_id)
        # the core's sync generator is pulled in a thread pool, events fed to the updater (async edge)
        aevents = iterate_in_threadpool(self._deep_search.stream(question))
        await _drive(updater, aevents)

    async def cancel(self, context, event_queue) -> None:  # pragma: no cover
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.failed()


def build_request_handler(deep_search, *, agent_card: AgentCard) -> DefaultRequestHandler:
    """A2A handler: our executor + in-memory task store + card (config-agnostic)."""
    return DefaultRequestHandler(
        agent_executor=RagAgentExecutor(deep_search),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )


def add_a2a_routes(
    app: FastAPI,
    deep_search,
    *,
    agent_card: AgentCard | None = None,
    rpc_url: str = DEFAULT_RPC_URL,
) -> FastAPI:
    """Mounts the Agent Card + JSON-RPC A2A routes on the same `app` (route factory). Returns app."""
    card = agent_card or build_agent_card()
    handler = build_request_handler(deep_search, agent_card=card)
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url),
    )
    return app
