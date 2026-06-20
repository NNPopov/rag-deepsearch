"""Search backends and fusion over ParadeDB (Rag_query_architecture.md §4).

Living here: VectorSearcher / BM25Searcher (psycopg, attached later), RRF, MMR, EXPAND.
The pure functions (rrf/mmr) do not depend on the DB and are tested as units; psycopg is pulled
in lazily inside the searcher classes, so importing the module does not require a live connection.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable

from rag.contracts import ExpandedSection, RetrievedChunk

ConnectFn = Callable[[], object]

# Reference-list sections: heading ends with References/Bibliography/Further Reading.
# Anchored to the tail — so that "Reference architecture" (content) is NOT caught; it catches
# "4.13.1 Caching references", "17.12 References". A heading heuristic → soft signal for a rank
# penalty (not removal).
_REFERENCES_RE = re.compile(r"(?i)(references|bibliography|further reading)\s*$")


def is_references_heading(heading: str | None) -> bool:
    """True if the section heading is a reference list (by its tail). None/empty → False."""
    return bool(heading) and bool(_REFERENCES_RE.search(heading))


def penalize_references(chunks: list[RetrievedChunk], penalty: float) -> list[RetrievedChunk]:
    """Lowers the RRF score of references chunks by the `penalty` multiplier (1.0 = no-op). Mutates in place.

    Does NOT remove and does NOT re-sort: a reference list with a strong raw score will still
    break through (a valid "give me the sources" query), while in an ordinary question it sinks
    below content. Sorting is the caller's job (the loop re-sorts the pool before MMR)."""
    if penalty == 1.0:
        return chunks
    for c in chunks:
        if c.is_references:
            c.score *= penalty
    return chunks


def _filter_doc_ids(filters: dict | None) -> list[int] | None:
    """Extracts document_ids from filters (narrowing search by document, §1/§4). None → no filter."""
    if filters and filters.get("document_ids"):
        return list(filters["document_ids"])
    return None

# RetrievedChunk columns in one SELECT — the shared row shape for both searchers.
# s.heading is needed for the references detector (rank penalty) — we lose nothing along the way.
_SELECT_COLS = (
    "c.chunk_id, c.content, c.section_id, c.document_id, "
    "d.title, s.path::text AS path, c.embedding, c.metadata, s.heading"
)
_N_COLS = 9  # how many columns _SELECT_COLS occupies (score/dist comes next)


def _row_to_chunk(row, *, score: float) -> RetrievedChunk:
    """DB row (_SELECT_COLS) + explicit score → RetrievedChunk. We lose nothing along the way."""
    chunk_id, content, section_id, document_id, title, path, embedding, metadata, heading = row
    source = f"{title} › {path}" if path else (title or "")
    vector = embedding.to_list() if embedding is not None else None  # halfvec → list (MMR)
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        document_id=document_id,
        section_id=section_id,
        source=source,
        score=score,
        vector=vector,
        is_references=is_references_heading(heading),
        metadata=metadata or {},
    )


class VectorSearcher:
    """KNN over chunks.embedding (HNSW cosine). Embeds the query via the injected gateway (§4).

    config-agnostic: the `connect` factory and `gateway` come via the constructor; it does not read config.
    Owns the connection lifecycle (open → close). score = 1 − cosine_distance —
    informative in standalone use; in the hybrid RRF overwrites it.
    """

    def __init__(self, *, connect: ConnectFn, gateway, top_k: int = 8) -> None:
        self._connect = connect
        self._gateway = gateway
        self._top_k = top_k

    async def search(self, query: str, *, filters: dict | None = None) -> list[RetrievedChunk]:
        out = await self.search_many([query], filters=filters)
        return out[0] if out else []

    async def search_many(
        self, queries: list[str], *, filters: dict | None = None
    ) -> list[list[RetrievedChunk]]:
        """Batch: ALL queries are embedded in ONE gateway call (§6), KNN over a single connection.

        Embedding is one network round-trip for N queries (instead of N sequential ones): the main
        SEARCH latency (see §6). The connection is also one for the whole batch — we do not churn
        connect/close per query. Returns a list of results in `queries` order."""
        from pgvector import HalfVector
        from pgvector.psycopg import register_vector_async

        queries = list(queries)
        if not queries:
            return []  # nothing to embed — we touch neither the gateway nor the DB

        vectors = await self._gateway.aembedding(texts=queries)   # ONE batch call for all queries
        doc_ids = _filter_doc_ids(filters)
        where = ""
        if doc_ids is not None:
            where = "WHERE c.document_id = ANY(%(doc_ids)s)"
        sql = f"""
            SELECT {_SELECT_COLS}, c.embedding <=> %(qv)s AS dist
            FROM chunks c
            JOIN documents d USING (document_id)
            LEFT JOIN sections s USING (section_id)
            {where}
            ORDER BY dist
            LIMIT %(k)s
        """
        results: list[list[RetrievedChunk]] = []
        conn = await self._connect()
        try:
            await register_vector_async(conn)
            for vec in vectors:
                params: dict = {"qv": HalfVector(vec), "k": self._top_k}
                if doc_ids is not None:
                    params["doc_ids"] = doc_ids
                cur = await conn.execute(sql, params)
                rows = await cur.fetchall()
                results.append(
                    [_row_to_chunk(r[:_N_COLS], score=1.0 - float(r[_N_COLS])) for r in rows]
                )
        finally:
            await conn.close()
        return results


class BM25Searcher:
    """pg_search full-text: content @@@ query + paradedb.score (§4). No gateway needed.

    Also pulls c.embedding → vector so MMR works over BM25 hits too (§5). score =
    paradedb.score (informative standalone; in the hybrid RRF overwrites it).
    """

    def __init__(self, *, connect: ConnectFn, top_k: int = 8) -> None:
        self._connect = connect
        self._top_k = top_k

    async def search(self, query: str, *, filters: dict | None = None) -> list[RetrievedChunk]:
        from pgvector.psycopg import register_vector_async

        params: dict = {"q": query, "k": self._top_k}
        extra = ""
        doc_ids = _filter_doc_ids(filters)
        if doc_ids is not None:
            extra = " AND c.document_id = ANY(%(doc_ids)s)"
            params["doc_ids"] = doc_ids
        conn = await self._connect()
        try:
            await register_vector_async(conn)  # needed to read c.embedding → vector
            cur = await conn.execute(
                f"""
                SELECT {_SELECT_COLS}, paradedb.score(c.chunk_id) AS score
                FROM chunks c
                JOIN documents d USING (document_id)
                LEFT JOIN sections s USING (section_id)
                WHERE c.content @@@ %(q)s{extra}
                ORDER BY score DESC
                LIMIT %(k)s
                """,
                params,
            )
            rows = await cur.fetchall()
        finally:
            await conn.close()
        return [_row_to_chunk(r[:_N_COLS], score=float(r[_N_COLS])) for r in rows]


class Expander:
    """EXPAND / small2big (§4): section_id of top chunks → ancestor block (chapter, depth=1) via ltree @>.

    Synthesis runs over the chapter's `full_text`, not over the trimmings of small chunks. Unique
    section_ids of the top chunks resolve to their ancestor chapter (`@>`, scoped by document_id,
    otherwise an identical path "1" would glue different documents together). Dedup by the chapter's
    section_id; `from_chunks` is provenance (which chunk_ids led into this block). config-agnostic:
    the `connect` factory comes via the constructor.
    """

    def __init__(self, *, connect: ConnectFn) -> None:
        self._connect = connect

    async def expand(self, chunks: list[RetrievedChunk]) -> list[ExpandedSection]:
        small_ids = list({c.section_id for c in chunks if c.section_id is not None})
        if not small_ids:
            return []  # nothing to expand — we do not touch the DB

        conn = await self._connect()
        try:
            cur = await conn.execute(
                """
                SELECT small.section_id AS small_id,
                       big.section_id, big.full_text, big.document_id, d.title, big.path::text
                FROM sections small
                JOIN sections big
                  ON big.document_id = small.document_id   -- otherwise path "1" glues different documents
                 AND big.path @> small.path                -- ancestor in ltree (including the chapter itself)
                 AND big.depth = 1                          -- Variant A: chapter
                JOIN documents d ON d.document_id = big.document_id
                WHERE small.section_id = ANY(%(ids)s)
                """,
                {"ids": small_ids},
            )
            rows = await cur.fetchall()
        finally:
            await conn.close()

        small_to_big: dict[int, tuple] = {}
        for small_id, big_id, full_text, doc_id, title, path in rows:
            source = f"{title} › {path}" if path else (title or "")
            small_to_big[small_id] = (big_id, full_text, doc_id, source)

        blocks: dict[int, ExpandedSection] = {}  # order = order of appearance of the top chunks
        for c in chunks:
            if c.section_id is None:
                continue
            mapped = small_to_big.get(c.section_id)
            if mapped is None:
                continue
            big_id, full_text, doc_id, source = mapped
            block = blocks.get(big_id)
            if block is None:
                block = ExpandedSection(
                    section_id=big_id, document_id=doc_id,
                    full_text=full_text, source=source, from_chunks=[],
                )
                blocks[big_id] = block
            block.from_chunks.append(c.chunk_id)
        return list(blocks.values())


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity of two vectors; None / zero vector → 0.0 (no similarity penalty)."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def mmr(
    candidates: list[RetrievedChunk], *, lambda_: float = 0.5, top_n: int = 12
) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance: a diversity re-rank before EXPAND (§5, problem B).

    Relevance = RRF score (chunk.score). Step: argmax [λ·rel − (1−λ)·max cos(c, selected)].
    The first is taken by max relevance (selected is empty → no similarity penalty).
    cosine is computed in Python over vector; fewer candidates than top_n → all are returned.
    """
    pool = list(candidates)
    selected: list[RetrievedChunk] = []
    while pool and len(selected) < top_n:
        if not selected:
            best = max(pool, key=lambda c: c.score)
        else:
            best = max(
                pool,
                key=lambda c: lambda_ * c.score
                - (1.0 - lambda_) * max(_cosine(c.vector, s.vector) for s in selected),
            )
        selected.append(best)
        pool.remove(best)
    return selected


def rrf(result_lists: list[list[RetrievedChunk]], *, k: int = 60) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: score = Σ 1/(k + rank), rank from 1 (as in S14, k=60).

    Fuses N vector + M bm25 ranked lists. Dedup by chunk_id (UNIQUE): one object per chunk,
    contributions from all lists are summed. Returns sorted by descending score;
    each RetrievedChunk gets its final RRF score set.
    """
    scores: dict[int, float] = {}
    chosen: dict[int, RetrievedChunk] = {}
    for lst in result_lists:
        for rank, chunk in enumerate(lst, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chosen.setdefault(chunk.chunk_id, chunk)   # the first object encountered per chunk_id

    for cid, chunk in chosen.items():
        chunk.score = scores[cid]
    return sorted(chosen.values(), key=lambda c: c.score, reverse=True)
