"""Shared db package: schema, connection, models, the contract dimension constant.

Public API:
- DIMENSION         — the contract embedding dimension (matches the halfvec(N) schema);
- schema.SCHEMA_SQL — the schema DDL (ParadeDB);
- schema.apply_schema(conn) — idempotent schema application.
"""

# The contract embedding dimension. Decision: text-embedding-3-large, dimensions=1536.
# CHANGE ONLY together with the schema (halfvec(N)) — the ingest and rag configs must match.
DIMENSION = 1536

__all__ = ["DIMENSION"]
