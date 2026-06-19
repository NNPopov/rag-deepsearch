"""R8 (red): Persist filter `ingest.persist.PersistStep` (S12) — load into live ParadeDB.

Contract §4.5 Rag_pipeline_architecture.md + `Rag_schema_sections.md`: read the section tree
(`02_structure/sections.jsonl`, `Section`) and embeddings (`04_embed/chunks_embedded.jsonl`,
`EmbeddedChunk`), resolve `document_id` (by `external_id`) and `section_id` (by `path`), load
into the `documents`/`sections`/`chunks` tables. IMPORTANT — idempotency: re-loading the same
data does not create duplicates (`ON CONFLICT (document_id, content_hash) DO NOTHING`).

The connection is INJECTED (config-agnostic `db`): the `connect` factory comes in the constructor.
The test runs against LIVE `rag_test` (the `schema_conn` fixture applies the schema); we isolate
the chunks with our own `external_id` so as not to depend on the rest of the database's content.
RED before S12: the ingest.persist module does not exist yet.
"""
from __future__ import annotations

import os

import pytest

import db
from ingest.contracts import Chunk, EmbeddedChunk, Section

EXT = "ddia-persist-r8"  # this test's isolating external_id

TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://rag:rag@localhost:55432/rag_test"
)


@pytest.fixture
def connect(schema_conn):
    """Factory of FRESH connections to rag_test (schema already applied by `schema_conn`).

    PersistStep owns the connection (open → commit → close); the test's assertions go
    through a separate `schema_conn`, so the step won't close the test's connection.
    """
    import psycopg

    def _factory():
        return psycopg.connect(TEST_DSN)

    return _factory


def _vec(first: float) -> list[float]:
    v = [0.0] * db.DIMENSION
    v[0] = first
    return v


def _sections() -> list[Section]:
    chapter = Section(
        external_id=EXT, path="1", depth=1, parent_path=None, ordinal=1,
        heading="Replication", is_leaf=False, text="big chapter text", token_count=3,
    )
    sub1 = Section(
        external_id=EXT, path="1.1", depth=2, parent_path="1", ordinal=1,
        heading="Single-Leader", is_leaf=True, text="leader based replication", token_count=3,
        page_range=(227, 227),
    )
    sub2 = Section(
        external_id=EXT, path="1.2", depth=2, parent_path="1", ordinal=2,
        heading="Multi-Leader", is_leaf=True, text="multi leader writes", token_count=3,
    )
    return [chapter, sub1, sub2]


def _embedded() -> list[EmbeddedChunk]:
    def ec(section_path: str, seq: int, text: str, first: float) -> EmbeddedChunk:
        base = Chunk(
            external_id=EXT, section_path=section_path, chapter=1, heading="H",
            seq=seq, text=text, token_count=3, metadata={"k": "v"},
        )
        return EmbeddedChunk(
            **base.model_dump(exclude={"id", "content_hash"}),
            vector=_vec(first), embedding_model="text-embedding-3-large",
        )

    return [
        ec("1.1", 0, "leader based replication and failover", 1.0),
        ec("1.1", 1, "synchronous versus asynchronous", 0.5),
        ec("1.2", 0, "conflict resolution in multi leader", 0.25),
    ]


def _counts(conn):
    n_doc = conn.execute(
        "SELECT count(*) FROM documents WHERE external_id = %s", (EXT,)
    ).fetchone()[0]
    doc_id = conn.execute(
        "SELECT document_id FROM documents WHERE external_id = %s", (EXT,)
    ).fetchone()
    if doc_id is None:
        return n_doc, 0, 0
    doc_id = doc_id[0]
    n_sec = conn.execute(
        "SELECT count(*) FROM sections WHERE document_id = %s", (doc_id,)
    ).fetchone()[0]
    n_chunk = conn.execute(
        "SELECT count(*) FROM chunks WHERE document_id = %s", (doc_id,)
    ).fetchone()[0]
    return n_doc, n_sec, n_chunk


# --- Step contract (S6) --------------------------------------------------------------------
def test_conforms_to_step_protocol():
    from ingest.orchestrator import Step
    from ingest.persist import PersistStep

    step = PersistStep(connect=lambda: None)
    assert isinstance(step, Step)
    assert step.name == "05_persist"


# --- load resolves document/section and writes all rows -----------------------------------
def test_persist_writes_documents_sections_chunks(schema_conn, connect):
    from ingest.persist import PersistStep

    step = PersistStep(connect=connect)
    step.persist(_sections(), _embedded())

    n_doc, n_sec, n_chunk = _counts(schema_conn)
    assert n_doc == 1
    assert n_sec == 3      # chapter + 2 subchapters
    assert n_chunk == 3    # 3 embeddings


def test_chunks_resolve_to_correct_section(schema_conn, connect):
    from ingest.persist import PersistStep

    PersistStep(connect=connect).persist(_sections(), _embedded())
    doc_id = schema_conn.execute(
        "SELECT document_id FROM documents WHERE external_id = %s", (EXT,)
    ).fetchone()[0]
    rows = schema_conn.execute(
        """
        SELECT s.path, count(*)
        FROM chunks c JOIN sections s ON s.section_id = c.section_id
        WHERE c.document_id = %s
        GROUP BY s.path ORDER BY s.path
        """,
        (doc_id,),
    ).fetchall()
    assert rows == [("1.1", 2), ("1.2", 1)]  # 2 chunks in 1.1, 1 in 1.2


# --- idempotency: a repeat does not create duplicates ----------------------------------------------
def test_persist_is_idempotent(schema_conn, connect):
    from ingest.persist import PersistStep

    step = PersistStep(connect=connect)
    step.persist(_sections(), _embedded())
    first = _counts(schema_conn)
    step.persist(_sections(), _embedded())  # repeat of the same data
    second = _counts(schema_conn)

    assert first == (1, 3, 3)
    assert second == (1, 3, 3)  # ON CONFLICT DO NOTHING → no duplicates


# --- small2big: chunk → chapter via ltree ancestor ------------------------------------------------
def test_small2big_after_persist(schema_conn, connect):
    from ingest.persist import PersistStep

    PersistStep(connect=connect).persist(_sections(), _embedded())
    doc_id = schema_conn.execute(
        "SELECT document_id FROM documents WHERE external_id = %s", (EXT,)
    ).fetchone()[0]
    chunk_id = schema_conn.execute(
        "SELECT chunk_id FROM chunks WHERE document_id = %s ORDER BY chunk_index LIMIT 1",
        (doc_id,),
    ).fetchone()[0]
    big = schema_conn.execute(
        """
        SELECT DISTINCT a.heading, a.full_text
        FROM chunks c
        JOIN sections s ON s.section_id = c.section_id
        JOIN sections a ON a.document_id = s.document_id AND a.path @> s.path AND a.depth = 1
        WHERE c.chunk_id = %s
        """,
        (chunk_id,),
    ).fetchall()
    assert big == [("Replication", "big chapter text")]


# --- works as an orchestrator step: reads both jsonl files, writes a receipt artifact ---------------
def test_runs_as_orchestrator_step(tmp_path, schema_conn, connect):
    from ingest.orchestrator import Orchestrator
    from ingest.persist import PersistStep
    from ingest.rundir import RunDir

    base = tmp_path / "runs"
    guid = "g-persist"
    rd = RunDir(base, guid)
    (rd.stage_dir("02_structure") / "sections.jsonl").write_text(
        "\n".join(s.model_dump_json() for s in _sections()) + "\n", encoding="utf-8"
    )
    (rd.stage_dir("04_embed") / "chunks_embedded.jsonl").write_text(
        "\n".join(e.model_dump_json() for e in _embedded()) + "\n", encoding="utf-8"
    )

    Orchestrator([PersistStep(connect=connect)], base=base).run(guid)

    assert rd.path("05_persist", "receipt.json").exists()
    _, n_sec, n_chunk = _counts(schema_conn)
    assert (n_sec, n_chunk) == (3, 3)


# --- the embedding makes it through: KNN finds the nearest chunk -----------------------------------------
def test_embedding_roundtrips_knn(schema_conn, connect):
    from pgvector import HalfVector

    from ingest.persist import PersistStep

    PersistStep(connect=connect).persist(_sections(), _embedded())
    doc_id = schema_conn.execute(
        "SELECT document_id FROM documents WHERE external_id = %s", (EXT,)
    ).fetchone()[0]
    top = schema_conn.execute(
        "SELECT content FROM chunks WHERE document_id = %s ORDER BY embedding <=> %s LIMIT 1",
        (doc_id, HalfVector(_vec(1.0))),
    ).fetchone()[0]
    assert "leader based replication" in top  # the first chunk's vector (first=1.0)
