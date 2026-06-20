"""rag CLI entry point (Q10): `python -m rag --question "..."` — adapter over `DeepSearch.run()`.

Thin wrapper over the composition root (`rag.app`): parse arguments, read config ONCE,
build `DeepSearch` via DI, run the question and print the answer. The CLI is just one of the
adapters over the same core (§3.3: SSE/A2A — part 3); there is no search logic here.

Override flags selectively override thresholds from config; `--document-ids` narrows search via a filter.
`load`/`build` are injected (tests substitute them to verify parsing without network/DB).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from rag.app import build_deep_search
from rag.config import load_settings


def _force_utf8() -> None:
    """Avoid crashing on Cyrillic/arrows in a cp1252 Windows console (output is UTF-8)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass  # stream is no longer a TextIOWrapper (redirected) — skip


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag",
        description="RAG deep-search: question -> answer over the corpus (PLAN-SEARCH-REFLECT-EXPAND-SYNTH).",
    )
    p.add_argument("--question", "-q", required=True, help="question to the corpus")
    p.add_argument(
        "--format", choices=["text", "json"], default="text", help="output format"
    )
    p.add_argument(
        "--trace",
        action="store_true",
        help="show retrieved chunks (source/score/fragment) — proof of grounding",
    )
    p.add_argument(
        "--document-ids", type=int, nargs="*", help="narrow the search to these document_id"
    )
    # selective threshold override flags (None → take the value from config)
    p.add_argument("--top-k", type=int, help="candidates per query")
    p.add_argument("--rrf-k", type=int, help="RRF constant")
    p.add_argument("--mmr-lambda", type=float, help="relevance↔diversity balance (MMR)")
    p.add_argument("--mmr-top-n", type=int, help="how many to keep after MMR")
    p.add_argument("--relevance-threshold", type=float, help="CRAG chunk-cutoff threshold")
    p.add_argument("--max-chunks", type=int, help="chunk budget before EXPAND")
    p.add_argument("--max-iterations", type=int, help="ceiling on REFLECT-loop iterations")
    p.add_argument("--cost-ceiling", type=float, help="query cost ceiling")
    p.add_argument(
        "--corpus-language", help="corpus language (search queries are translated into it)"
    )
    p.add_argument(
        "--references-penalty", type=float,
        help="rank multiplier for bibliography-list sections (1.0 = off)",
    )
    return p


def _overrides_from_args(args) -> dict:
    """Builds the CLI override dict (None values are filtered out by build_deep_search)."""
    return {
        "top_k": args.top_k,
        "rrf_k": args.rrf_k,
        "mmr_lambda": args.mmr_lambda,
        "mmr_top_n": args.mmr_top_n,
        "relevance_threshold": args.relevance_threshold,
        "max_chunks": args.max_chunks,
        "max_iterations": args.max_iterations,
        "cost_ceiling": args.cost_ceiling,
        "corpus_language": args.corpus_language,
        "references_penalty": args.references_penalty,
    }


def _snippet(text: str, limit: int = 300) -> str:
    """Collapse whitespace runs and truncate to limit — a fragment for the trace."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"


def _render_trace(result) -> str:
    """Evidence base: exactly which chunks were retrieved from the corpus and fed into synthesis.

    Prints the granular hits (`result.chunks` — what actually matched the query, with the
    RRF score), so any claim in the answer can be checked against its source."""
    lines = ["", "─" * 60, f"TRACE: {len(result.chunks)} retrieved chunks "
             f"(REFLECT iterations: {result.iterations})"]
    for c in sorted(result.chunks, key=lambda c: c.score, reverse=True):
        lines.append("")
        lines.append(f"[{c.source}]  score={c.score:.4f}  chunk_id={c.chunk_id}")
        lines.append(f"  {_snippet(c.content)}")
    return "\n".join(lines)


def _render(result, fmt: str, *, trace: bool = False) -> str:
    if fmt == "json":
        return result.model_dump_json(indent=2)
    lines = [result.answer, "", "Sources:"]
    lines += [f"- {c}" for c in result.citations]
    out = "\n".join(lines)
    if trace:
        out += "\n" + _render_trace(result)
    return out


def _run_async(coro):
    """Drive an async coroutine to completion from the sync CLI entry (ADR-0002: the core is async).

    On Windows psycopg's AsyncConnection cannot run on the default ProactorEventLoop — it needs a
    SelectorEventLoop; we select it for this process before asyncio.run."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coro)


def main(argv: list[str] | None = None, *, load=load_settings, build=build_deep_search) -> int:
    _force_utf8()
    args = build_parser().parse_args(argv)

    settings = load()  # the single config read (composition root)
    deep_search = build(settings, overrides=_overrides_from_args(args))

    filters = {"document_ids": args.document_ids} if args.document_ids else None
    result = _run_async(deep_search.run(args.question, filters=filters))
    print(_render(result, args.format, trace=args.trace))
    return 0


if __name__ == "__main__":
    sys.exit(main())
