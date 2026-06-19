"""ingest pipe contracts (Pydantic) — one model per jsonl line.

Source of truth: §3 Rag_pipeline_architecture.md. Validation at every boundary catches
contract drift immediately. The pipe models are internal to ingest; DB table rows live in the `db` package.

Three models and transitions:
  Structure → Chunk : `Section`         (02_structure/sections.jsonl)
  Chunk → Embed     : `Chunk`           (03_chunk/chunks.jsonl)
  Embed → Persist   : `EmbeddedChunk`   (04_embed/chunks_embedded.jsonl)

Identity rules:
  • id = "{external_id}:{section_path}:{seq}" — deterministic business id;
  • content_hash = sha256(id + "\\x00" + text) — id is mixed in so identical text
    in different sections doesn't collapse (unique index (document_id, content_hash));
  • page_range — best-effort (None allowed), NOT a boundary.
"""
import hashlib

from pydantic import BaseModel, Field, computed_field, field_validator

import db

PageRange = tuple[int, int]


class Section(BaseModel):
    """Structure tree node → `sections` row. A leaf (`is_leaf`) goes to Chunk."""

    external_id: str
    path: str  # ltree, e.g. "1.2" (chapter.subchapter)
    depth: int  # 1 = chapter, 2 = subchapter
    parent_path: str | None = None
    ordinal: int
    heading: str
    is_leaf: bool
    text: str  # full section text (full_text in the DB, big-block for small2big)
    token_count: int
    page_range: PageRange | None = None  # approximate; None for a page-less format
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """A chunk (Chunk → Embed). `id`/`content_hash` are derived — computed from the fields."""

    external_id: str
    section_path: str
    chapter: int
    heading: str
    seq: int
    text: str
    token_count: int
    page_range: PageRange | None = None  # carried through from the section
    metadata: dict = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def id(self) -> str:
        return f"{self.external_id}:{self.section_path}:{self.seq}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        return hashlib.sha256((self.id + "\x00" + self.text).encode("utf-8")).hexdigest()


class EmbeddedChunk(Chunk):
    """A chunk with a vector (Embed → Persist). Vector length = the contract db.DIMENSION."""

    vector: list[float]
    embedding_model: str  # provenance (e.g. text-embedding-3-large)

    @field_validator("vector")
    @classmethod
    def _vector_len_matches_contract(cls, v: list[float]) -> list[float]:
        if len(v) != db.DIMENSION:
            raise ValueError(
                f"vector length {len(v)} != db.DIMENSION ({db.DIMENSION}); "
                "must match the halfvec(N) schema."
            )
        return v
