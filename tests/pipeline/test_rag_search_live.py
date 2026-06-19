"""QR3/QR6 — live searchers (rag.search.VectorSearcher / BM25Searcher) against rag_test.

Basis: Rag_query_architecture.md §4. The hybrid is assembled from two backends over ParadeDB:
  • VectorSearcher — embeds the query via the gateway, KNN over chunks.embedding (HNSW cosine);
  • BM25Searcher — content @@@ query + paradedb.score (pg_search), without a gateway.
Both return a FULL RetrievedChunk: chunk_id/content/section_id/document_id/source/vector —
nothing is lost along the way (this is the input to RRF/MMR). Seed — TWO documents (multi-doc seam §5.⚠):
docA about replication, docB about partitioning; this checks that document_id distinguishes
sources and that top_k caps the output.

External-dependency strategy (CLAUDE.md §5): we MOCK the query embedding (no network),
the DB itself is a live rag_test (the schema_conn fixture applies the schema). RED before QR3: no searchers.
"""
from __future__ import annotations

import os

import pytest

import db
from rag.contracts import RetrievedChunk

TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://rag:rag@localhost:55432/rag_test"
)

A = "rag-search-A"
B = "rag-search-B"


def _vec(*pairs: tuple[int, float]) -> list[float]:
    """Sparse vector of length DIMENSION: we set (index, value) pairs, the rest is 0."""
    v = [0.0] * db.DIMENSION
    for i, val in pairs:
        v[i] = val
    return v


class FakeGateway:
    """Gateway mock: embedding(texts=...) → a pre-set vector (no network)."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.calls: list[tuple[list[str], str]] = []

    def embedding(self, *, texts, task: str = "embed") -> list[list[float]]:
        self.calls.append((list(texts), task))
        return [list(self._vector) for _ in texts]


@pytest.fixture
def connect(schema_conn):
    """Factory of FRESH connections to rag_test (the schema was already applied by schema_conn)."""
    import psycopg

    def _factory():
        return psycopg.connect(TEST_DSN)

    return _factory


@pytest.fixture
def seeded(schema_conn):
    """Seeds TWO documents into rag_test. schema_conn autocommit → visible to fresh connections.

    Idempotent (ON CONFLICT + a repeated SELECT id): rag_test survives between session tests.
    """
    from pgvector import HalfVector
    from pgvector.psycopg import register_vector

    conn = schema_conn
    register_vector(conn)

    def upsert_doc(ext: str, title: str) -> int:
        conn.execute(
            "INSERT INTO documents (external_id, title, source) VALUES (%s, %s, %s) "
            "ON CONFLICT (external_id) DO UPDATE SET title = EXCLUDED.title",
            (ext, title, ext),
        )
        return conn.execute(
            "SELECT document_id FROM documents WHERE external_id = %s", (ext,)
        ).fetchone()[0]

    def upsert_section(doc_id: int, path: str, heading: str, full_text: str, depth: int = 1) -> int:
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

    # IMPORTANT: vocabulary and vector indices do NOT overlap with test_persist (R8) in the same rag_test —
    # that one writes "…replication/leader…" with a vector at index 0; we use unique words and dim 10/20.
    da = upsert_doc(A, "Doc A")
    sa = upsert_section(da, "1", "Caching", "caching chapter full text")
    sa_sub = upsert_section(da, "1.1", "Eviction", "eviction subsection text", depth=2)  # for small2big
    upsert_chunk(da, sa, 0, "cache invalidation and eviction policies", _vec((10, 1.0)))
    upsert_chunk(da, sa, 1, "writethrough caching strategy", _vec((10, 0.9), (11, 0.1)))

    dbid = upsert_doc(B, "Doc B")
    sb = upsert_section(dbid, "1", "Partitioning", "partitioning chapter full text")
    upsert_chunk(dbid, sb, 0, "partitioning splits keys by range", _vec((20, 1.0)))
    upsert_chunk(dbid, sb, 1, "rebalancing partitions across nodes", _vec((20, 0.9), (21, 0.1)))

    return {"A": da, "B": dbid, "sa": sa, "sa_sub": sa_sub, "sb": sb}


# --- QR3: VectorSearcher (KNN, query embedding via the gateway) ----------------------

def test_vector_searcher_carries_full_chunk(seeded, connect):
    from rag.search import VectorSearcher

    gw = FakeGateway(_vec((10, 1.0)))  # query "like" docA chunk0
    out = VectorSearcher(connect=connect, gateway=gw, top_k=4).search("cache eviction")

    assert gw.calls and gw.calls[0][0] == ["cache eviction"]  # the query went to embedding
    assert out, "vector search returned nothing"
    top = out[0]
    assert isinstance(top, RetrievedChunk)
    assert top.content.startswith("cache invalidation")          # nearest by cosine
    assert top.document_id == seeded["A"]
    assert top.section_id == seeded["sa"]                          # small2big key not lost
    assert "Doc A" in top.source                                  # source = title › path
    assert top.vector is not None and len(top.vector) == db.DIMENSION  # for MMR


def test_vector_searcher_respects_top_k(seeded, connect):
    from rag.search import VectorSearcher

    gw = FakeGateway(_vec((10, 1.0)))
    out = VectorSearcher(connect=connect, gateway=gw, top_k=2).search("x")
    assert len(out) == 2


def test_vector_searcher_batches_embeddings_in_one_call(seeded, connect):
    """§6: all of a round's vector queries are embedded in ONE batch gateway call, not one by one.

    The embeddings API accepts an array of inputs → one HTTP round-trip for N queries instead of N
    sequential ones. search_many is the batch entry: it returns a result for EACH query.
    """
    from rag.search import VectorSearcher

    gw = FakeGateway(_vec((10, 1.0)))
    vs = VectorSearcher(connect=connect, gateway=gw, top_k=2)
    lists = vs.search_many(["cache eviction", "writethrough caching"])

    assert len(gw.calls) == 1                                   # ONE embedding call for all queries
    assert gw.calls[0][0] == ["cache eviction", "writethrough caching"]
    assert len(lists) == 2                                      # a result for each query
    assert all(isinstance(c, RetrievedChunk) for lst in lists for c in lst)
    assert all(len(lst) <= 2 for lst in lists)                 # top_k respected for each


def test_vector_searcher_search_many_empty_skips_gateway(connect):
    """Empty input → empty output, we do not call the gateway (nothing to embed)."""
    from rag.search import VectorSearcher

    gw = FakeGateway(_vec((10, 1.0)))
    out = VectorSearcher(connect=connect, gateway=gw).search_many([])
    assert out == []
    assert gw.calls == []


def test_vector_searcher_search_many_filters_by_document(seeded, connect):
    """The document filter is forwarded to each query of the batch."""
    from rag.search import VectorSearcher

    gw = FakeGateway(_vec((10, 1.0)))
    lists = VectorSearcher(connect=connect, gateway=gw, top_k=4).search_many(
        ["cache", "eviction"], filters={"document_ids": [seeded["B"]]}
    )
    assert lists and all(c.document_id == seeded["B"] for lst in lists for c in lst)


def test_vector_searcher_distinguishes_documents(seeded, connect):
    from rag.search import VectorSearcher

    gw = FakeGateway(_vec((20, 1.0)))  # query "like" docB
    out = VectorSearcher(connect=connect, gateway=gw, top_k=2).search("partition")
    assert {c.document_id for c in out} == {seeded["B"]}  # top-2 from docB


def test_vector_searcher_filters_by_document(seeded, connect):
    from rag.search import VectorSearcher

    gw = FakeGateway(_vec((10, 1.0)))  # semantics closer to docA…
    out = VectorSearcher(connect=connect, gateway=gw, top_k=4).search(
        "cache", filters={"document_ids": [seeded["B"]]}
    )
    assert out and all(c.document_id == seeded["B"] for c in out)  # …but the filter narrowed to docB


# --- QR6: BM25Searcher (content @@@ query, paradedb.score) ---------------------------

def test_bm25_searcher_finds_keyword(seeded, connect):
    from rag.search import BM25Searcher

    out = BM25Searcher(connect=connect, top_k=4).search("partitioning")
    assert out, "bm25 returned nothing"
    assert all(c.document_id == seeded["B"] for c in out)  # "partition…" only in docB
    top = out[0]
    assert "partition" in top.content.lower()
    assert "Doc B" in top.source
    assert top.score > 0  # paradedb.score is set


def test_bm25_searcher_carries_ids_and_section(seeded, connect):
    from rag.search import BM25Searcher

    out = BM25Searcher(connect=connect, top_k=4).search("caching")
    assert out
    top = out[0]
    assert top.document_id == seeded["A"]
    assert top.section_id == seeded["sa"]
    assert top.chunk_id > 0


def test_bm25_searcher_respects_top_k(seeded, connect):
    from rag.search import BM25Searcher

    out = BM25Searcher(connect=connect, top_k=1).search("caching")
    assert len(out) == 1
    assert out[0].document_id == seeded["A"]


def test_bm25_searcher_empty_when_no_match(seeded, connect):
    from rag.search import BM25Searcher

    out = BM25Searcher(connect=connect, top_k=4).search("kubernetes")
    assert out == []  # the term is not in the corpus → honestly empty (no fallback)


def test_bm25_searcher_filters_by_document(seeded, connect):
    from rag.search import BM25Searcher

    # "partitioning" exists only in docB; filter to docA → empty
    out = BM25Searcher(connect=connect, top_k=4).search(
        "partitioning", filters={"document_ids": [seeded["A"]]}
    )
    assert out == []


# --- QR6 (Q5): EXPAND / small2big (section_id of top chunks → chapter block via ltree @>) -

def test_expand_maps_chunk_to_parent_chapter(seeded, connect):
    from rag.search import Expander

    leaf = RetrievedChunk(
        chunk_id=901, content="x", document_id=seeded["A"],
        section_id=seeded["sa_sub"], source="s",   # leaf of subchapter 1.1
    )
    blocks = Expander(connect=connect).expand([leaf])
    assert len(blocks) == 1
    b = blocks[0]
    assert b.section_id == seeded["sa"]                 # chapter depth=1 (ancestor of 1.1)
    assert b.full_text == "caching chapter full text"   # synthesis over the chapter's full_text, not a truncation
    assert b.document_id == seeded["A"]
    assert "Doc A" in b.source
    assert b.from_chunks == [901]                       # provenance


def test_expand_dedups_sections_and_collects_provenance(seeded, connect):
    from rag.search import Expander

    # chapter "1" (itself depth=1) and its subchapter "1.1" → ONE chapter block, provenance collected
    c1 = RetrievedChunk(chunk_id=1, content="x", document_id=seeded["A"],
                        section_id=seeded["sa"], source="s")
    c2 = RetrievedChunk(chunk_id=2, content="y", document_id=seeded["A"],
                        section_id=seeded["sa_sub"], source="s")
    blocks = Expander(connect=connect).expand([c1, c2])
    assert len(blocks) == 1
    assert blocks[0].section_id == seeded["sa"]
    assert sorted(blocks[0].from_chunks) == [1, 2]


def test_expand_spans_multiple_documents(seeded, connect):
    from rag.search import Expander

    ca = RetrievedChunk(chunk_id=1, content="x", document_id=seeded["A"],
                        section_id=seeded["sa"], source="s")
    cb = RetrievedChunk(chunk_id=2, content="y", document_id=seeded["B"],
                        section_id=seeded["sb"], source="s")
    blocks = Expander(connect=connect).expand([ca, cb])
    assert {b.document_id for b in blocks} == {seeded["A"], seeded["B"]}  # @> scoped by document


def test_expand_ignores_null_section(seeded, connect):
    from rag.search import Expander

    c = RetrievedChunk(chunk_id=1, content="x", document_id=seeded["A"],
                       section_id=None, source="s")
    assert Expander(connect=connect).expand([c]) == []


def test_expand_empty_input(connect):
    from rag.search import Expander

    assert Expander(connect=connect).expand([]) == []
