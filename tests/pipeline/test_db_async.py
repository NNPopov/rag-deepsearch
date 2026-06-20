"""ADR-0002 L1 (red): the ASYNC db surface — `aconnect` + `afetchall` over psycopg AsyncConnection.

ADR-0002: `db` becomes dual-surface — the sync path stays (ingest persist), and an async connect/query
path is added for the query+serve core. The helpers are config-agnostic (CLAUDE.md §2): the DSN is
passed in, db reads no env/Dynaconf. They are the async analogue of how `rag.search` connects/queries
today (psycopg.connect → conn.execute(...).fetchall()).

External-dependency strategy (CLAUDE.md §5): live `rag_test` (the `schema_conn` fixture applies the
schema over a sync autocommit connection; a fresh AsyncConnection sees it). RED before L1: db exports
no aconnect/afetchall.

pytest-asyncio asyncio_mode=auto → `async def test_*` run without a per-test marker.
"""
from __future__ import annotations

import os

import pytest

TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://rag:rag@localhost:55432/rag_test"
)


# --- aconnect: opens a real AsyncConnection ------------------------------------------
async def test_aconnect_opens_async_connection(schema_conn):
    import psycopg

    from db import aconnect

    conn = await aconnect(TEST_DSN)
    try:
        assert isinstance(conn, psycopg.AsyncConnection)
        cur = await conn.execute("SELECT 1")
        assert (await cur.fetchone())[0] == 1
    finally:
        await conn.close()


async def test_aconnect_bad_dsn_raises(schema_conn):
    from db import aconnect

    with pytest.raises(Exception):  # noqa: B017 — any psycopg connection error is acceptable
        await aconnect("postgresql://rag:rag@localhost:55432/does_not_exist_db")


# --- afetchall: execute + fetchall over an AsyncConnection ----------------------------
async def test_afetchall_returns_all_rows(schema_conn):
    from db import aconnect, afetchall

    conn = await aconnect(TEST_DSN)
    try:
        rows = await afetchall(conn, "SELECT generate_series(1, 3) AS n")
        assert [r[0] for r in rows] == [1, 2, 3]
    finally:
        await conn.close()


async def test_afetchall_passes_params(schema_conn):
    from db import aconnect, afetchall

    conn = await aconnect(TEST_DSN)
    try:
        rows = await afetchall(
            conn, "SELECT %(a)s + %(b)s AS s", {"a": 40, "b": 2}
        )
        assert rows[0][0] == 42
    finally:
        await conn.close()


async def test_afetchall_reads_schema_table(schema_conn):
    """Sanity: the async path sees the schema applied by the (sync) schema_conn fixture."""
    from db import aconnect, afetchall

    conn = await aconnect(TEST_DSN)
    try:
        rows = await afetchall(conn, "SELECT count(*) FROM documents")
        assert rows[0][0] >= 0  # table exists and is queryable over the async connection
    finally:
        await conn.close()
