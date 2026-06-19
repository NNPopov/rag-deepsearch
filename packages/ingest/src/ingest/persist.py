"""Persist filter (S12) — loads the section tree and embeddings into ParadeDB.

Single responsibility (§4.5 Rag_pipeline_architecture.md, `Rag_schema_sections.md`): read
`02_structure/sections.jsonl` (`Section`, the whole tree) and `04_embed/chunks_embedded.jsonl`
(`EmbeddedChunk`, leaves with vectors), resolve `document_id` (by `external_id`) and `section_id`
(by `path`) and load the rows into `documents` / `sections` / `chunks`. The receipt artifact
`05_persist/receipt.json` lets the orchestrator see the step as done on restart.

Idempotency — re-loading the same data does NOT produce duplicates:
  • documents: `ON CONFLICT (external_id) DO UPDATE` (always returns document_id);
  • sections : `ON CONFLICT (document_id, path) DO NOTHING`;
  • chunks   : `ON CONFLICT (document_id, content_hash) DO NOTHING` (dedup on re-load).

DI / config-agnostic: the connection comes via the `connect()` factory (composition root, S13) — the step
owns its lifecycle (open → commit → close). The step itself doesn't read DSN/config; the `db` package
provides only the schema/`DIMENSION`. `chunk_index` is a per-document running ordinal (file order).
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from ingest.contracts import EmbeddedChunk, Section
from ingest.orchestrator import StepContext

ConnectFn = Callable[[], object]


class PersistStep:
    name: str = "05_persist"
    artifacts: Sequence[str] = ("05_persist/receipt.json",)

    def __init__(self, *, connect: ConnectFn, sections_stage: str = "02_structure") -> None:
        self._connect = connect
        self._sections_stage = sections_stage  # tree input: Structure, or Clean (S9) if enabled

    def params(self) -> dict:
        # The section source affects the loaded full_text (small2big) → into the hash for restart invalidation.
        return {"sections_stage": self._sections_stage}

    # --- orchestrator step ------------------------------------------------------
    def run(self, ctx: StepContext) -> str:
        sections = [
            Section.model_validate_json(ln)
            for ln in ctx.rundir.path(self._sections_stage, "sections.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if ln.strip()
        ]
        chunks = [
            EmbeddedChunk.model_validate_json(ln)
            for ln in ctx.rundir.path("04_embed", "chunks_embedded.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if ln.strip()
        ]
        receipt = self.persist(sections, chunks)
        out = ctx.stage_dir / "receipt.json"
        out.write_text(json.dumps(receipt, ensure_ascii=False), encoding="utf-8")
        return str(out)

    # --- load (live db; connection injected) --------------------------
    def persist(
        self, sections: Sequence[Section], chunks: Sequence[EmbeddedChunk]
    ) -> dict:
        from pgvector import HalfVector
        from pgvector.psycopg import register_vector
        from psycopg.types.json import Json

        external_id = self._external_id(sections, chunks)
        conn = self._connect()
        try:
            register_vector(conn)
            doc_id = conn.execute(
                """
                INSERT INTO documents (external_id, source)
                VALUES (%s, %s)
                ON CONFLICT (external_id) DO UPDATE SET updated_at = now()
                RETURNING document_id
                """,
                (external_id, external_id),
            ).fetchone()[0]

            # sections: the whole tree (chapters + subchapters) — needed for small2big via ltree
            for s in sections:
                conn.execute(
                    """
                    INSERT INTO sections
                        (document_id, path, depth, ordinal, heading, full_text, token_count, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, path) DO NOTHING
                    """,
                    (doc_id, s.path, s.depth, s.ordinal, s.heading, s.text,
                     s.token_count, Json(s.metadata)),
                )
            path_to_id = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT path::text, section_id FROM sections WHERE document_id = %s",
                    (doc_id,),
                ).fetchall()
            }
            # explicit parent_id link (convenience for JOIN/cascade; the ltree ancestor is available anyway)
            for s in sections:
                if s.parent_path and s.parent_path in path_to_id:
                    conn.execute(
                        "UPDATE sections SET parent_id = %s WHERE section_id = %s",
                        (path_to_id[s.parent_path], path_to_id[s.path]),
                    )

            # chunks: leaves with vectors. chunk_index — running per-document (file order)
            for chunk_index, c in enumerate(chunks):
                conn.execute(
                    """
                    INSERT INTO chunks
                        (document_id, section_id, chunk_index, content, embedding,
                         token_count, content_hash, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, content_hash)
                        WHERE content_hash IS NOT NULL DO NOTHING
                    """,
                    (
                        doc_id,
                        path_to_id.get(c.section_path),
                        chunk_index,
                        c.text,
                        HalfVector(c.vector),
                        c.token_count,
                        c.content_hash,
                        Json({**c.metadata, "embedding_model": c.embedding_model}),
                    ),
                )
            conn.commit()
        finally:
            conn.close()

        return {
            "external_id": external_id,
            "document_id": doc_id,
            "sections": len(sections),
            "chunks": len(chunks),
        }

    @staticmethod
    def _external_id(
        sections: Sequence[Section], chunks: Sequence[EmbeddedChunk]
    ) -> str:
        ids = {s.external_id for s in sections} | {c.external_id for c in chunks}
        if len(ids) != 1:
            raise ValueError(f"expected one external_id per run, got: {sorted(ids)}")
        return ids.pop()
