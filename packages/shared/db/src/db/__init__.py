"""Shared db package: schema, connection, models, the contract dimension constant.

Public API:
- DIMENSION         — the contract embedding dimension (matches the halfvec(N) schema);
- schema.SCHEMA_SQL — the schema DDL (ParadeDB);
- schema.apply_schema(conn) — idempotent schema application.
- aconnect(dsn)     — open a psycopg AsyncConnection (ADR-0002: async query+serve core);
- afetchall(conn, sql, params) — execute + fetchall over an AsyncConnection.

The async helpers are config-agnostic (CLAUDE.md §2): the DSN is passed in, db reads no
env/Dynaconf. They are the async analogue of how `rag.search` connects/queries today.
"""

# The contract embedding dimension. Decision: text-embedding-3-large, dimensions=1536.
# CHANGE ONLY together with the schema (halfvec(N)) — the ingest and rag configs must match.
DIMENSION = 1536

__all__ = ["DIMENSION", "aconnect", "afetchall"]


async def aconnect(dsn: str):
    """Open a real psycopg AsyncConnection to `dsn` (caller owns the lifecycle → await close)."""
    import psycopg  # noqa: PLC0415 — pulled lazily; importing db must not require psycopg

    return await psycopg.AsyncConnection.connect(dsn)


async def afetchall(conn, sql: str, params=None):
    """execute + fetchall over an AsyncConnection — the async analogue of conn.execute(...).fetchall()."""
    cur = await conn.execute(sql, params)
    return await cur.fetchall()
