"""R1 (red): DB schema on live ParadeDB.

Pins the `db` package contract (S1):
  • DIMENSION == 1536 (not the placeholder 1024 from the DDL);
  • DDL applies from scratch; chunks.embedding = halfvec(1536);
  • vector: insert + KNN search via the <=> operator;
  • BM25: full-text search via the @@@ operator (pg_search);
  • small2big: lift chunk → chapter via ltree ancestor (@>).

All tests are RED before S1: the db package has no DIMENSION/schema yet.
"""
import psycopg
import pytest


def test_dimension_is_1536():
    from db import DIMENSION

    assert DIMENSION == 1536


def test_embedding_column_is_halfvec_1536(schema_conn):
    from db import DIMENSION

    # type format in pg: "halfvec(1536)" — take it from the column's format_type attribute
    row = schema_conn.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        WHERE a.attrelid = 'chunks'::regclass
          AND a.attname = 'embedding'
        """
    ).fetchone()
    assert row is not None, "chunks.embedding column not found"
    assert row[0] == f"halfvec({DIMENSION})"


def test_required_indexes_exist(schema_conn):
    names = {
        r[0]
        for r in schema_conn.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'"
        ).fetchall()
    }
    assert "chunks_embedding_hnsw_idx" in names, "no pgvector HNSW index"
    assert "chunks_bm25_idx" in names, "no pg_search BM25 index"


def _seed_minimal(conn):
    """Document + one leaf section (chapter) + two chunks with embeddings."""
    from db import DIMENSION
    from pgvector import HalfVector

    doc_id = conn.execute(
        "INSERT INTO documents (source) VALUES ('test://ddia') RETURNING document_id"
    ).fetchone()[0]
    sec_id = conn.execute(
        """
        INSERT INTO sections (document_id, path, depth, ordinal, heading, full_text)
        VALUES (%s, '1', 1, 1, 'Replication', 'big chapter text')
        RETURNING section_id
        """,
        (doc_id,),
    ).fetchone()[0]

    base = [0.0] * DIMENSION
    near = base.copy(); near[0] = 1.0
    far = base.copy(); far[1] = 1.0
    conn.execute(
        """
        INSERT INTO chunks (document_id, section_id, chunk_index, content, embedding, content_hash)
        VALUES (%s, %s, 0, 'leader based replication and failover', %s, 'h0'),
               (%s, %s, 1, 'unrelated kitchen recipe', %s, 'h1')
        """,
        (doc_id, sec_id, HalfVector(near), doc_id, sec_id, HalfVector(far)),
    )
    return doc_id, sec_id


def test_vector_knn_search(schema_conn):
    from db import DIMENSION
    from pgvector import HalfVector

    _seed_minimal(schema_conn)
    q = [0.0] * DIMENSION
    q[0] = 1.0  # closer to the first chunk
    top = schema_conn.execute(
        "SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 1",
        (HalfVector(q),),
    ).fetchone()[0]
    assert "replication" in top


def test_bm25_search(schema_conn):
    _seed_minimal(schema_conn)
    rows = schema_conn.execute(
        "SELECT content FROM chunks WHERE content @@@ 'replication' "
        "ORDER BY paradedb.score(chunk_id) DESC"
    ).fetchall()
    assert any("replication" in r[0] for r in rows)
    assert all("recipe" not in r[0] for r in rows)


def test_small2big_ltree(schema_conn):
    doc_id, sec_id = _seed_minimal(schema_conn)
    chunk_id = schema_conn.execute(
        "SELECT chunk_id FROM chunks WHERE chunk_index = 0"
    ).fetchone()[0]
    big = schema_conn.execute(
        """
        SELECT DISTINCT a.heading, a.full_text
        FROM chunks c
        JOIN sections s ON s.section_id = c.section_id
        JOIN sections a ON a.document_id = s.document_id
                       AND a.path @> s.path
                       AND a.depth = 1
        WHERE c.chunk_id = ANY(%s::bigint[])
        """,
        ([chunk_id],),
    ).fetchall()
    assert big == [("Replication", "big chapter text")]
