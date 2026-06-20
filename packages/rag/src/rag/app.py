"""Query-side composition root (Q10) — the single place that reads config + assembles DI.

§6 Rag_query_architecture.md (mirrors `ingest/app.py`): config (`load_settings()`) is read
ONCE in the CLI, values flow down ONLY through constructors. The shared `gateway`/`connect`
are built here and injected into the searchers / expander / loop — they are config-agnostic
themselves (they do not read env/Dynaconf). `DeepSearch` steps do not call each other; the loop
itself enforces the order.

Assembly is lazy: no network, no DB at build time. litellm is invoked only when a step actually
calls the gateway; psycopg — when a searcher calls `connect()`. `gateway`/`connect` can be passed
from outside (tests/substitution); by default they are built from `settings`.

`overrides` — selective CLI overrides of thresholds (None values are ignored): the single
source remains settings, the flags merely override specific keys on top of it.
"""
from __future__ import annotations

from collections.abc import Callable

from rag.loop import DeepSearch
from rag.search import BM25Searcher, Expander, VectorSearcher


def build_model_map(settings) -> dict[str, str]:
    """"task → model" mapping for the gateway: PLAN/REFLECT/SYNTH + the query embedder (task='embed')."""
    return {
        "plan": settings.llm.plan,
        "reflect": settings.llm.reflect,
        "synth": settings.llm.synth,
        "embed": settings.embed.model,
    }


def build_gateway(
    settings, *, completion_fn=None, embedding_fn=None, acompletion_fn=None, aembedding_fn=None
):
    """The single LLM gateway from config (config-agnostic — values are passed to the constructor).

    Dual-surface (ADR-0002): the query side uses the async methods (`acompletion`/`aembedding`),
    which default to `litellm.acompletion`/`aembedding`; the sync methods are kept for ingest. The
    `*_fn` are injectable for tests/substitution."""
    from llm_gateway import Gateway

    return Gateway(
        model_map=build_model_map(settings),
        dimension=int(settings.embed.dimensions),
        retries=int(settings.llm.retries),
        completion_fn=completion_fn,
        embedding_fn=embedding_fn,
        acompletion_fn=acompletion_fn,
        aembedding_fn=aembedding_fn,
    )


def build_connect(settings) -> Callable[[], object]:
    """Factory of fresh ASYNC DB connections (ADR-0002): the searchers do `await connect()`.

    DSN is read lazily (on call, not at build time); `db.aconnect` opens a psycopg AsyncConnection."""

    async def _connect():
        from db import aconnect

        return await aconnect(str(settings.database_url))

    return _connect


def build_deep_search(
    settings,
    *,
    gateway=None,
    connect: Callable[[], object] | None = None,
    overrides: dict | None = None,
) -> DeepSearch:
    """Builds `DeepSearch` from settings via DI: searchers/expander get `connect`/`gateway`,
    the loop gets the thresholds. `overrides` overrides values from settings (None values are skipped)."""
    gateway = gateway if gateway is not None else build_gateway(settings)
    connect = connect if connect is not None else build_connect(settings)

    params: dict = {
        "top_k": int(settings.search.top_k),
        "rrf_k": int(settings.search.rrf_k),
        "mmr_lambda": float(settings.search.mmr_lambda),
        "mmr_top_n": int(settings.search.mmr_top_n),
        "relevance_threshold": float(settings.search.relevance_threshold),
        "max_chunks": int(settings.search.max_chunks),
        "max_iterations": int(settings.loop.max_iterations),
        "cost_ceiling": float(settings.loop.cost_ceiling),
        "corpus_language": str(settings.corpus.language),  # cross-lingual search (§5)
        "references_penalty": float(settings.search.references_penalty),  # rank penalty for reference lists
    }
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})

    top_k = params.pop("top_k")  # top_k is a searcher parameter, not a loop one
    vector = VectorSearcher(connect=connect, gateway=gateway, top_k=top_k)
    bm25 = BM25Searcher(connect=connect, top_k=top_k)
    expander = Expander(connect=connect)
    return DeepSearch(
        vector_searcher=vector,
        bm25_searcher=bm25,
        expander=expander,
        gateway=gateway,
        **params,
    )
