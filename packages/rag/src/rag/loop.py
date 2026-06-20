"""Deep-search loop — the query-side core (Rag_query_architecture.md §1, §3, §5, §6).

`DeepSearch` orchestrates PLAN → SEARCH(hybrid RRF) → REFLECT(source-aware) → LOOP →
EXPAND(small2big) → MMR → SYNTH. Web-free and ASYNC (ADR-0002): the I/O steps await the async
gateway/searchers/expander, while the pure transforms (RRF/MMR/`_cited_sources`) stay synchronous
— coloring a function that never awaits would be contagion for nothing.

Two entry points over ONE event generator (§3.3): `stream()` emits an `Event` stream (for
transport), `run()` drives that same `stream()` and returns the `DeepSearchResult` from the
final `Final`. The generator is the single source of truth, so SSE/A2A and the CLI see an identical run.

DI (§6, CLAUDE.md §2): the searchers / expander / gateway and the thresholds come ONLY via the
constructor; the loop itself does not read config (the composition root `rag/app.py` reads it).
`steps.plan/reflect/synth` are called directly — they too are config-agnostic and go to the same injected gateway.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable

from rag import steps
from rag.contracts import (
    AnswerDelta,
    DeepSearchResult,
    Event,
    ExpandReady,
    Final,
    PlanReady,
    ReflectResult,
    RetrievedChunk,
    SearchRound,
)
from rag.search import mmr, penalize_references, rrf


def _unique(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")          # contents of each [...] in the answer
_NUM_RE = re.compile(r"\d+(?:\.\d+)*")               # chapter/section numbers (3, 12, 1.4.6)


def _bracket_matches_chapter(bracket_low: str, chapter: str) -> bool:
    """The block's chapter occurs in the bracket. Numeric chapter → compare the FIRST component
    of the number (1.4.6 → chapter 1), so the model's deeper reference still counts; a non-numeric
    path (fixtures 'A'/'B') → substring."""
    if not chapter:
        return False
    if chapter.isdigit():
        return any(n.split(".", 1)[0] == chapter for n in _NUM_RE.findall(bracket_low))
    return chapter in bracket_low


def _cited_sources(answer: str, block_sources: list[str]) -> list[str]:
    """Which block_sources are ACTUALLY cited in the answer — robust to the label format.

    The model is supposed to cite as `[title › path]`, but in practice it changes the separator
    (`[title, 4.8]`) and deepens the number (4.8 instead of chapter 4) — then an exact match
    `[s] in answer` misses and the source list falls back to "all blocks" (dishonest). Here we
    match on two signals WITHIN a single bracket: (1) the book title is contained in the bracket;
    (2) the block's chapter matches the number in the bracket. We only consider the contents of
    `[...]` — a mention of "chapter 8" in prose is not a citation. We return the CANONICAL
    block_sources (not how the model wrote them), in block order of appearance."""
    brackets_low = [m.group(1).lower() for m in _BRACKET_RE.finditer(answer)]
    if not brackets_low:
        return []
    cited: list[str] = []
    for source in block_sources:
        title, _, path = source.partition(" › ")
        title_low = title.strip().lower()
        chapter = path.strip().split(".", 1)[0].lower() if path else ""
        if title_low and any(
            title_low in b and _bracket_matches_chapter(b, chapter) for b in brackets_low
        ):
            cited.append(source)
    return cited


class DeepSearch:
    def __init__(
        self,
        *,
        vector_searcher,
        bm25_searcher,
        expander,
        gateway,
        rrf_k: int = 60,
        mmr_lambda: float = 0.5,
        mmr_top_n: int = 12,
        relevance_threshold: float = 0.0,
        max_chunks: int = 40,
        max_iterations: int = 3,
        cost_ceiling: float = 0.0,
        cost_fn: Callable[[], float] | None = None,
        corpus_language: str = steps.DEFAULT_CORPUS_LANGUAGE,
        references_penalty: float = 1.0,
    ) -> None:
        self._vector = vector_searcher
        self._bm25 = bm25_searcher
        self._expander = expander
        self._gateway = gateway
        self._rrf_k = rrf_k
        self._mmr_lambda = mmr_lambda
        self._mmr_top_n = mmr_top_n
        self._relevance_threshold = relevance_threshold
        self._max_chunks = max_chunks
        self._max_iterations = max_iterations
        self._cost_ceiling = cost_ceiling
        self._cost_fn = cost_fn
        self._corpus_language = corpus_language  # corpus language → queries are translated into it (§5)
        self._references_penalty = references_penalty  # rank penalty for references sections (1.0=off)

    # --- public interface ------------------------------------------------------------
    async def run(self, question: str, *, filters: dict | None = None) -> DeepSearchResult | None:
        """Drives stream() to the end and returns the result from the final Final (§3.3)."""
        result: DeepSearchResult | None = None
        async for ev in self.stream(question, filters=filters):
            if isinstance(ev, Final):
                result = ev.result
        return result

    async def stream(self, question: str, *, filters: dict | None = None) -> AsyncIterator[Event]:
        trace: list[Event] = []

        def emit(ev: Event) -> Event:
            trace.append(ev)
            return ev

        # --- PLAN --------------------------------------------------------------------
        plan = await steps.plan(question, gateway=self._gateway, corpus_language=self._corpus_language)
        yield emit(PlanReady(plan=plan))

        # --- SEARCH / REFLECT / LOOP -------------------------------------------------
        accumulated: dict[int, RetrievedChunk] = {}   # dedup by chunk_id (§4 RRF)
        vqs = list(plan.vector_queries)
        bqs = list(plan.bm25_queries)
        iteration = 0
        while True:
            iteration += 1

            # SEARCH: N vector + M bm25 lists → RRF → dedup accumulation.
            # All vector queries of the round in ONE batch embedding (§6); bm25 is lexical, no gateway.
            lists = await self._vector.search_many(vqs, filters=filters)
            for q in bqs:
                lists.append(await self._bm25.search(q, filters=filters))
            fused = rrf(lists, k=self._rrf_k)
            penalize_references(fused, self._references_penalty)  # reference lists — down, not out
            new_count = 0
            for c in fused:
                prev = accumulated.get(c.chunk_id)
                if prev is None:
                    accumulated[c.chunk_id] = c
                    new_count += 1
                elif c.score > prev.score:        # keep the best RRF score per chunk
                    prev.score = c.score
            yield emit(SearchRound(
                iteration=iteration, vector_queries=vqs, bm25_queries=bqs,
                new_chunks=new_count, total_chunks=len(accumulated),
            ))

            # the re-query brought nothing → the corpus is exhausted for these formulations (§5 CRAG)
            if iteration > 1 and new_count == 0:
                break

            # REFLECT (source-aware): the critic sees ALL accumulated results
            reflection = await steps.reflect(
                question, plan, list(accumulated.values()),
                gateway=self._gateway, relevance_threshold=self._relevance_threshold,
                corpus_language=self._corpus_language,
            )
            yield emit(ReflectResult(
                iteration=iteration, is_sufficient=reflection.is_sufficient, gaps=reflection.gaps,
            ))

            if reflection.is_sufficient:
                break
            if iteration >= self._max_iterations:
                break
            new_vqs = reflection.new_vector_queries
            new_bqs = reflection.new_bm25_queries
            if not (new_vqs or new_bqs):          # gaps exist but no queries → stop
                break
            if self._cost_ceiling > 0 and self._cost_fn and self._cost_fn() >= self._cost_ceiling:
                break
            vqs, bqs = new_vqs, new_bqs

        # --- EXPAND (MMR diversity → small2big) --------------------------------------
        pool = [c for c in accumulated.values() if c.score >= self._relevance_threshold]
        pool.sort(key=lambda c: c.score, reverse=True)
        pool = pool[: self._max_chunks]
        selected = mmr(pool, lambda_=self._mmr_lambda, top_n=self._mmr_top_n)
        blocks = await self._expander.expand(selected)
        yield emit(ExpandReady(
            section_ids=[b.section_id for b in blocks], blocks=len(blocks),
        ))

        # --- SYNTH (stream tokens over the blocks' full_text) ------------------------
        parts: list[str] = []
        async for tok in steps.synth_stream(question, blocks, gateway=self._gateway):
            parts.append(tok)
            yield emit(AnswerDelta(text=tok))
        answer = "".join(parts)

        block_sources = _unique([b.source for b in blocks])
        cited = _cited_sources(answer, block_sources)
        result = DeepSearchResult(
            answer=answer,
            citations=cited or block_sources,
            chunks=selected,
            blocks=blocks,
            plan=plan,
            iterations=iteration,
            trace=list(trace),                    # without Final — otherwise self-recursion (§3.3)
            cost=self._cost_fn() if self._cost_fn else None,
        )
        yield Final(result=result)
