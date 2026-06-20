"""Shared fixtures for RAG tests.

The test DB `rag_test` lives in the SAME ParadeDB container (book-rag-db),
reachable from the host at localhost:55432. We never touch the working `rag` DB.

Overridable via env:
  TEST_MAINT_DSN     — DSN to the maintenance DB for CREATE/DROP DATABASE (default .../rag)
  TEST_DATABASE_URL  — DSN to the test DB itself (default .../rag_test)
"""
import asyncio
import os
import selectors
import sys

import pytest


def pytest_asyncio_loop_factories(config, item):
    """ADR-0002: psycopg's AsyncConnection cannot run on Windows' default ProactorEventLoop —
    it needs a SelectorEventLoop. pytest-asyncio builds each test's loop from this factory.
    Off Windows we contribute nothing (return None) → the default loop is used.
    """
    if sys.platform == "win32":
        return {"selector": lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())}
    return None

MAINT_DSN = os.environ.get(
    "TEST_MAINT_DSN", "postgresql://rag:rag@localhost:55432/rag"
)
TEST_DB = "rag_test"
TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://rag:rag@localhost:55432/rag_test"
)


@pytest.fixture(scope="session")
def test_database():
    """Recreates an empty rag_test DB for the test session and drops it at the end."""
    import psycopg

    with psycopg.connect(MAINT_DSN, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        conn.execute(f"CREATE DATABASE {TEST_DB}")
    try:
        yield TEST_DSN
    finally:
        with psycopg.connect(MAINT_DSN, autocommit=True) as conn:
            conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")


@pytest.fixture
def schema_conn(test_database):
    """Connection to rag_test with the schema applied (db.schema.apply_schema).

    RED until S1 is implemented: importing db.schema / apply_schema will fail.
    """
    import psycopg
    from pgvector.psycopg import register_vector

    from db.schema import apply_schema  # noqa: PLC0415 — intentionally lazy import

    with psycopg.connect(test_database, autocommit=True) as conn:
        apply_schema(conn)
        register_vector(conn)
        yield conn
