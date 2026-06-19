"""DB schema DDL (ParadeDB) + idempotent application to a connection.

The source of truth for the structure is `Rag_schema_sections.md`. Here is the same schema, but:
- `halfvec(__DIM__)` is substituted from the contract constant `DIMENSION` (not a placeholder 1024);
- HNSW index on `halfvec_cosine_ops`;
- all objects are created with `IF NOT EXISTS` (application is idempotent — the test fixture runs it
  before every test against the same rag_test database).

Query examples (vector/BM25/RRF/small2big) live in `Rag_schema_sections.md`, not here —
that is application code, not schema.
"""
from __future__ import annotations

from db import DIMENSION

# We substitute the dimension via .replace("__DIM__", ...), NOT f-string/format —
# the BM25 index DDL has JSON curly braces that f-string/format would eat.
_DDL = r"""
-- 0. Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;
CREATE EXTENSION IF NOT EXISTS ltree;

-- 1. Documents (source level)
CREATE TABLE IF NOT EXISTS documents (
    document_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    external_id   TEXT UNIQUE,
    title         TEXT,
    source        TEXT NOT NULL,
    uri           TEXT,
    mime_type     TEXT,
    metadata      JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_metadata_gin ON documents USING gin (metadata jsonb_path_ops);

-- 2. Section tree (ltree: chapters / subchapters / ...)
CREATE TABLE IF NOT EXISTS sections (
    section_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id   BIGINT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    parent_id     BIGINT REFERENCES sections(section_id) ON DELETE CASCADE,
    path          ltree  NOT NULL,
    depth         INT    NOT NULL,
    ordinal       INT    NOT NULL,
    heading       TEXT,
    full_text     TEXT,
    token_count   INT,
    metadata      JSONB  NOT NULL DEFAULT '{}',
    UNIQUE (document_id, path)
);
CREATE INDEX IF NOT EXISTS sections_path_gist       ON sections USING gist (path);
CREATE INDEX IF NOT EXISTS sections_document_id_idx ON sections (document_id);
CREATE INDEX IF NOT EXISTS sections_parent_id_idx   ON sections (parent_id);

-- 3. Chunks (the retrieval unit). embedding halfvec(N) — N = DIMENSION.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id   BIGINT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    section_id    BIGINT REFERENCES sections(section_id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,
    content       TEXT NOT NULL,
    embedding     halfvec(__DIM__),
    token_count   INT,
    content_hash  TEXT,
    metadata      JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_section_id_idx  ON chunks (section_id);
CREATE INDEX IF NOT EXISTS chunks_metadata_gin    ON chunks USING gin (metadata jsonb_path_ops);
CREATE UNIQUE INDEX IF NOT EXISTS chunks_doc_hash_uidx ON chunks (document_id, content_hash) WHERE content_hash IS NOT NULL;

-- 4. Vector index (pgvector HNSW, cosine)
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64);

-- 5. BM25 index (ParadeDB pg_search), English with stemming
CREATE INDEX IF NOT EXISTS chunks_bm25_idx ON chunks USING bm25 (chunk_id, content, metadata) WITH (key_field = 'chunk_id', text_fields = '{ "content": { "tokenizer": { "type": "default", "stemmer": "English" }, "record": "position" } }')
"""

SCHEMA_SQL = _DDL.replace("__DIM__", str(DIMENSION))


def _split_statements(sql: str) -> list[str]:
    """Splits the DDL into individual statements.

    psycopg3 works over the extended protocol and does NOT execute multiple commands in a single
    execute(), so we run them one at a time. Comment lines (`--`) are dropped. Our DDL has
    no ';' inside string literals/comments, so split on ';' is safe.
    """
    statements = []
    for raw in sql.split(";"):
        body = "\n".join(
            line for line in raw.splitlines() if not line.strip().startswith("--")
        ).strip()
        if body:
            statements.append(body)
    return statements


def apply_schema(conn) -> None:
    """Idempotently applies the schema to an open psycopg3 connection."""
    for statement in _split_statements(SCHEMA_SQL):
        conn.execute(statement)
