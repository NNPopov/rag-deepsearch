"""Composition root (S13) — the single place that reads config + assembles the pipe via DI.

§7 Rag_pipeline_architecture.md: config (`load_settings()`) is read ONCE here, values flow
down ONLY through constructors. Shared `gateway`/`connect` are built here and injected into
the steps — they are themselves config-agnostic (don't read env/Dynaconf). Steps don't call
each other; the `Orchestrator` holds the order and the run.

Assembly is lazy: no network, no DB at build time. litellm is touched only when EmbedStep
actually calls the gateway; psycopg — when PersistStep calls `connect()`. That's why
`source`/`external_id` (which describe a concrete document, not a setting) come as arguments,
not from config.

`gateway`/`connect` can be passed in from outside (tests/substitution); by default built from `settings`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from ingest.chunk import ChunkStep
from ingest.clean import CleanStep
from ingest.embed import EmbedStep
from ingest.extract import ExtractStep
from ingest.orchestrator import Orchestrator, Step
from ingest.persist import PersistStep
from ingest.structure import StructureStep


def build_model_map(settings) -> dict[str, str]:
    """Task → model mapping for the gateway. EmbedStep uses task='embed'; mechanics use 'default'."""
    return {
        "embed": settings.embed.model,
        "default": settings.llm.default,
    }


def build_gateway(settings, *, completion_fn=None, embedding_fn=None):
    """The single LLM gateway from config (config-agnostic — values handed to the constructor)."""
    from llm_gateway import Gateway

    return Gateway(
        model_map=build_model_map(settings),
        dimension=settings.embed.dimensions,
        retries=settings.llm.retries,
        completion_fn=completion_fn,
        embedding_fn=embedding_fn,
    )


def build_connect(settings) -> Callable[[], object]:
    """Factory of fresh DB connections (DSN from config, read lazily — at call time)."""

    def _connect():
        import psycopg

        return psycopg.connect(str(settings.database_url))

    return _connect


def build_steps(
    settings,
    *,
    source,
    external_id: str,
    gateway=None,
    connect: Callable[[], object] | None = None,
    chapter_pattern: str | None = None,
    subsection_mode: str | None = None,
    opener_pattern: str | None = None,
    clean: bool = False,
    clean_method: str = "normalize",
) -> list[Step]:
    """Assembles the pipe steps in order, threading values from settings (DI).

    `chapter_pattern`/`subsection_mode`/`opener_pattern` — per-book structure profile (§4.2.1);
    None → StructureStep default (ddia/footers/caps opener). `clean=True` inserts optional Clean (S9)
    AFTER Structure and routes the Chunk+Persist input to `03_clean` (otherwise both read `02_structure`).
    `clean_method`: `normalize` (deterministic default) / `llm` (via gateway).
    """
    gateway = gateway if gateway is not None else build_gateway(settings)
    connect = connect if connect is not None else build_connect(settings)
    structure_kwargs: dict = {}
    if chapter_pattern:
        structure_kwargs["chapter_pattern"] = chapter_pattern
    if subsection_mode:
        structure_kwargs["subsection_mode"] = subsection_mode
    if opener_pattern is not None:
        structure_kwargs["opener_pattern"] = opener_pattern

    sections_stage = "03_clean" if clean else "02_structure"
    steps: list[Step] = [
        ExtractStep(source=source),
        StructureStep(external_id=external_id, **structure_kwargs),
    ]
    if clean:
        steps.append(CleanStep(method=clean_method, gateway=gateway))
    steps += [
        ChunkStep(
            chunk_size=settings.chunk.chunk_size,
            overlap=settings.chunk.overlap,
            sections_stage=sections_stage,
        ),
        EmbedStep(
            gateway=gateway,
            embedding_model=settings.embed.model,
            max_batch_tokens=settings.embed.max_batch_tokens,
            max_batch_size=settings.embed.max_batch_size,
        ),
        PersistStep(connect=connect, sections_stage=sections_stage),
    ]
    return steps


def build_orchestrator(
    settings,
    *,
    source,
    external_id: str,
    gateway=None,
    connect: Callable[[], object] | None = None,
    chapter_pattern: str | None = None,
    subsection_mode: str | None = None,
    opener_pattern: str | None = None,
    clean: bool = False,
    clean_method: str = "normalize",
) -> Orchestrator:
    """Assembles the orchestrator for the whole pipe. `run_dir` from config — the `runs/` root."""
    steps: Sequence[Step] = build_steps(
        settings,
        source=source,
        external_id=external_id,
        gateway=gateway,
        connect=connect,
        chapter_pattern=chapter_pattern,
        subsection_mode=subsection_mode,
        opener_pattern=opener_pattern,
        clean=clean,
        clean_method=clean_method,
    )
    return Orchestrator(steps, base=settings.run_dir)
