"""Query-side contracts — Pydantic models at the boundaries (Rag_query_architecture.md §3).

Three groups by origin:
  • from the DB (the searcher builds it, we trust it — we check the shape): RetrievedChunk, ExpandedSection;
  • from the LLM (we parse JSON, we MUST validate/reject malformed data): SearchPlan, Reflection;
  • stream/result: Event (union, discriminated by "type"), DeepSearchResult.

Parsing LLM output goes through Model.model_validate_json(...): malformed/incomplete JSON surfaces
as a ValidationError (caught in the step, logged into the trace) rather than leaking "silent" garbage downstream.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


# --- 3.1 From the DB -----------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """A chunk from search. Carries section_id/document_id/vector — loses nothing (cf. a thin sketch)."""

    chunk_id: int                       # chunks.chunk_id — UNIQUE → dedup key in RRF
    content: str                        # chunks.content
    document_id: int                    # key for source-aware REFLECT
    section_id: int | None              # small2big key (may be NULL)
    source: str                         # "{documents.title} › {sections.path}" — assembled via join
    score: float = 0.0                  # RRF score; set during fusion, NOT from the DB
    vector: list[float] | None = None   # chunks.embedding (halfvec→list) — NEEDED by MMR
    is_references: bool = False          # reference-list section (heading) → rank penalty, not removal
    metadata: dict = Field(default_factory=dict)  # page_start/end — for citations


class ExpandedSection(BaseModel):
    """EXPAND result (small2big): the large ancestor block (chapter, depth=1)."""

    section_id: int                     # sections.section_id of the large block
    document_id: int
    full_text: str                      # sections.full_text — material for SYNTH
    source: str                         # human-readable source for the citation
    from_chunks: list[int] = Field(default_factory=list)  # which chunk_ids led here (provenance)


# --- 3.2 From the LLM (validated) ----------------------------------------------------

def _clean(values: list[str]) -> list[str]:
    """strip + drop empties — shared cleanup of query lists from the LLM."""
    return [s.strip() for s in values if s and s.strip()]


class SearchPlan(BaseModel):
    """PLAN output: atomic sub-questions + expanded/keyword queries."""

    sub_questions: list[str] = Field(min_length=1)   # 2–5 atomic sub-questions
    vector_queries: list[str]           # expanded reformulations (semantic)
    bm25_queries: list[str]             # short keyword queries

    @field_validator("sub_questions", "vector_queries", "bm25_queries")
    @classmethod
    def _strip_empty(cls, v: list[str]) -> list[str]:
        return _clean(v)

    @field_validator("bm25_queries")  # runs last → sees the already-cleaned fields
    @classmethod
    def _require_some_query(cls, v: list[str], info) -> list[str]:
        if not (info.data.get("vector_queries") or v):
            raise ValueError("SearchPlan requires at least one vector_query or bm25_query")
        return v


class DocumentCoverage(BaseModel):
    """Coverage of sub-questions by a single document (source-aware REFLECT)."""

    document: str                       # human-readable label (title/external_id)
    covered: list[str] = []             # sub-questions covered by THIS document
    missing: list[str] = []             # what was not found in it


class Reflection(BaseModel):
    """REFLECT output (source-aware). is_sufficient=False && gaps && no new_* → loop stops."""

    is_sufficient: bool
    answered: list[str] = []            # closed sub-questions (across the whole corpus)
    gaps: list[str] = []                # what was not found — specifically
    coverage_by_document: list[DocumentCoverage] = []
    new_vector_queries: list[str] = []
    new_bm25_queries: list[str] = []


# --- 3.3 Stream and result -----------------------------------------------------------

class PlanReady(BaseModel):
    type: Literal["plan_ready"] = "plan_ready"
    plan: SearchPlan


class SearchRound(BaseModel):
    type: Literal["search_round"] = "search_round"
    iteration: int
    vector_queries: list[str]
    bm25_queries: list[str]
    new_chunks: int
    total_chunks: int


class ReflectResult(BaseModel):
    type: Literal["reflect"] = "reflect"
    iteration: int
    is_sufficient: bool
    gaps: list[str]


class ExpandReady(BaseModel):
    type: Literal["expand_ready"] = "expand_ready"
    section_ids: list[int]
    blocks: int


class AnswerDelta(BaseModel):
    type: Literal["answer_delta"] = "answer_delta"
    text: str                           # SYNTH SSE tokens


class Final(BaseModel):
    type: Literal["final"] = "final"
    result: DeepSearchResult


Event = Annotated[                       # discriminated by the "type" field
    PlanReady | SearchRound | ReflectResult | ExpandReady | AnswerDelta | Final,
    Field(discriminator="type"),
]


class DeepSearchResult(BaseModel):
    answer: str
    citations: list[str] = []            # sources the answer cited
    chunks: list[RetrievedChunk] = []    # accumulated used chunks
    blocks: list[ExpandedSection] = []   # large blocks fed into SYNTH
    plan: SearchPlan
    iterations: int
    trace: list[Event] = []              # the full event stream
    cost: float | None = None            # total token cost (if the gateway logs it)


# Final references DeepSearchResult declared below → finalize the references.
Final.model_rebuild()
