"""ADR-0002 L2 (red): async searchers/EXPAND — async port of QR3/QR6 (test_rag_search_live.py).

L2 flips `rag.search` I/O to async: VectorSearcher/BM25Searcher/search_many/Expander.expand →
`async def` over an async `connect` factory (`conn = await connect()`, `await conn.execute(...)`,
`await cur.fetchall()`, `await conn.close()`) and `await gateway.aembedding(...)`. The pure functions
RRF/MMR stay SYNC (their tests test_rag_rrf.py/test_rag_mmr.py are unchanged) — coloring a function
that never awaits would be contagion for nothing (ADR-0002 §3).

External-dependency strategy (CLAUDE.md §5): live `rag_test` (seeded over the sync `schema_conn`
fixture; the async searcher opens its OWN AsyncConnection and reads it back). The query embedding is
MOCKED via an ASYNC fake gateway (no network). RED before L2: the searchers are synchronous.

asyncio_mode=auto → async tests need no per-test marker.
"""
from __future__ import annotations

import os

import pytest

import db
from rag.contracts import RetrievedChunk

TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://rag:rag@localhost:55432/rag_test"
)

A = "rag-search-async-A"
B = "rag-search-async-B"


def _vec(*pairs: tuple[int, float]) -> list[float]:
    v = [0.0] * db.DIMENSION
    for i, val in pairs:
        v[i] = val
    return v


class FakeGateway:
    """Async gateway mock: aembedding(texts=...) → a pre-set vector (no network)."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.calls: list[tuple[list[str], str]] = []

    async def aembedding(self, *, texts, task: str = "embed") -> list[list[float]]:
        self.calls.append((list(texts), task))
        return [list(self._vector) for _ in texts]


@pytest.fixture
def connect(schema_conn):
    """Async factory of FRESH AsyncConnections to rag_test (schema already applied by schema_conn)."""
    import psycopg

    async def _factory():
        return await psycopg.AsyncConnection.connect(TEST_DSN)

    return _factory


@pytest.fixture
def seeded(schema_conn):
    """Seeds TWO documents into rag_test over the sync connection (autocommit → visible to async)."""
    from pgvector import HalfVector

    conn = schema_conn  # register_vector already applied by the fixture

    def upsert_doc(ext: str, title: str) -> int:
        conn.execute(
            "INSERT INTO documents (external_id, title, source) VALUES (%s, %s, %s) "
            "ON CONFLICT (external_id) DO UPDATE SET title = EXCLUDED.title",
            (ext, title, ext),
        )
        return conn.execute(
            "SELECT document_id FROM documents WHERE external_id = %s", (ext,)
        ).fetchone()[0]

    def upsert_section(doc_id, path, heading, full_text, depth=1) -> int:
        conn.execute(
            "INSERT INTO sections (document_id, path, depth, ordinal, heading, full_text) "
            "VALUES (%s, %s, %s, 1, %s, %s) ON CONFLICT (document_id, path) DO NOTHING",
            (doc_id, path, depth, heading, full_text),
        )
        return conn.execute(
            "SELECT section_id FROM sections WHERE document_id = %s AND path = %s::ltree",
            (doc_id, path),
        ).fetchone()[0]

    def upsert_chunk(doc_id, sec_id, idx, content, vector) -> None:
        conn.execute(
            "INSERT INTO chunks (document_id, section_id, chunk_index, content, embedding, token_count) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (document_id, chunk_index) DO NOTHING",
            (doc_id, sec_id, idx, content, HalfVector(vector), 5),
        )

    # Unique vocabulary + vector indices (40/41/50/51) — must NOT collide with the sync search seed
    # (10/11/20/21) or test_persist (index 0) in the shared rag_test.
    da = upsert_doc(A, "Doc A async")
    sa = upsert_section(da, "1", "Sharding", "sharding chapter full text")
    sa_sub = upsert_section(da, "1.1", "Resharding", "resharding subsection text", depth=2)
    upsert_chunk(da, sa, 100, "sharding distributes documents across shards", _vec((40, 1.0)))
    upsert_chunk(da, sa, 101, "resharding rebalances shards online", _vec((40, 0.9), (41, 0.1)))

    dbid = upsert_doc(B, "Doc B async")
    sb = upsert_section(dbid, "1", "Consensus", "consensus chapter full text")
    upsert_chunk(dbid, sb, 100, "consensus algorithms agree on a value", _vec((50, 1.0)))
    upsert_chunk(dbid, sb, 101, "quorum reads and writes for consensus", _vec((50, 0.9), (51, 0.1)))

    return {"A": da, "B": dbid, "sa": sa, "sa_sub": sa_sub, "sb": sb}


# --- QR3 (async): VectorSearcher -----------------------------------------------------

async def test_vector_searcher_carries_full_chunk(seeded, connect):
    from rag.search import VectorSearcher

    gw = FakeGateway(_vec((40, 1.0)))
    out = await VectorSearcher(connect=connect, gateway=gw, top_k=4).search("sharding")

    assert gw.calls and gw.calls[0][0] == ["sharding"]
    assert out, "vector search returned nothing"
    top = out[0]
    assert isinstance(top, RetrievedChunk)
    assert top.content.startswith("sharding distributes")
    assert top.document_id == seeded["A"]
    assert top.section_id == seeded["sa"]
    assert "Doc A async" in top.source
    assert top.vector is not None and len(top.vector) == db.DIMENSION


async def test_vector_searcher_respects_top_k(seeded, connect):
    from rag.search import VectorSearcher

    gw = FakeGateway(_vec((40, 1.0)))
    out = await VectorSearcher(connect=connect, gateway=gw, top_k=2).search("x")
    assert len(out) == 2


async def test_vector_searcher_batches_embeddings_in_one_call(seeded, connect):
    from rag.search import VectorSearcher

    gw = FakeGateway(_vec((40, 1.0)))
    vs = VectorSearcher(connect=connect, gateway=gw, top_k=2)
    lists = await vs.search_many(["sharding", "resharding"])

    assert len(gw.calls) == 1                                   # ONE async embedding call for all queries
    assert gw.calls[0][0] == ["sharding", "resharding"]
    assert len(lists) == 2
    assert all(isinstance(c, RetrievedChunk) for lst in lists for c in lst)
    assert all(len(lst) <= 2 for lst in lists)


async def test_vector_searcher_search_many_empty_skips_gateway(connect):
    from rag.search import VectorSearcher

    gw = FakeGateway(_vec((40, 1.0)))
    out = await VectorSearcher(connect=connect, gateway=gw).search_many([])
    assert out == []
    assert gw.calls == []


async def test_vector_searcher_filters_by_document(seeded, connect):
    from rag.search import VectorSearcher

    gw = FakeGateway(_vec((40, 1.0)))  # semantically closer to docA…
    out = await VectorSearcher(connect=connect, gateway=gw, top_k=4).search(
        "shard", filters={"document_ids": [seeded["B"]]}
    )
    assert out and all(c.document_id == seeded["B"] for c in out)  # …but narrowed to docB


# --- QR6 (async): BM25Searcher -------------------------------------------------------

async def test_bm25_searcher_finds_keyword(seeded, connect):
    from rag.search import BM25Searcher

    out = await BM25Searcher(connect=connect, top_k=4).search("consensus")
    assert out, "bm25 returned nothing"
    assert all(c.document_id == seeded["B"] for c in out)
    top = out[0]
    assert "consensus" in top.content.lower()
    assert "Doc B async" in top.source
    assert top.score > 0


async def test_bm25_searcher_respects_top_k(seeded, connect):
    from rag.search import BM25Searcher

    out = await BM25Searcher(connect=connect, top_k=1).search("sharding")
    assert len(out) == 1
    assert out[0].document_id == seeded["A"]


async def test_bm25_searcher_empty_when_no_match(seeded, connect):
    from rag.search import BM25Searcher

    out = await BM25Searcher(connect=connect, top_k=4).search("kubernetes")
    assert out == []


async def test_bm25_searcher_filters_by_document(seeded, connect):
    from rag.search import BM25Searcher

    out = await BM25Searcher(connect=connect, top_k=4).search(
        "consensus", filters={"document_ids": [seeded["A"]]}
    )
    assert out == []


# --- QR6 (async): EXPAND / small2big -------------------------------------------------

async def test_expand_maps_chunk_to_parent_chapter(seeded, connect):
    from rag.search import Expander

    leaf = RetrievedChunk(
        chunk_id=9901, content="x", document_id=seeded["A"],
        section_id=seeded["sa_sub"], source="s",
    )
    blocks = await Expander(connect=connect).expand([leaf])
    assert len(blocks) == 1
    b = blocks[0]
    assert b.section_id == seeded["sa"]
    assert b.full_text == "sharding chapter full text"
    assert b.document_id == seeded["A"]
    assert b.from_chunks == [9901]


async def test_expand_spans_multiple_documents(seeded, connect):
    from rag.search import Expander

    ca = RetrievedChunk(chunk_id=1, content="x", document_id=seeded["A"],
                        section_id=seeded["sa"], source="s")
    cb = RetrievedChunk(chunk_id=2, content="y", document_id=seeded["B"],
                        section_id=seeded["sb"], source="s")
    blocks = await Expander(connect=connect).expand([ca, cb])
    assert {b.document_id for b in blocks} == {seeded["A"], seeded["B"]}


async def test_expand_empty_input(connect):
    from rag.search import Expander

    assert await Expander(connect=connect).expand([]) == []
