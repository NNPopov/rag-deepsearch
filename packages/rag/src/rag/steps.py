"""Deep-search loop steps that go through the LLM gateway (Rag_query_architecture.md §1, §3, §5).

PLAN / REFLECT / SYNTH are the query-side "filters": each takes input, calls the INJECTED
gateway and returns a contract. config-agnostic (the gateway comes as an argument; they do not read
config, thresholds come as a parameter). LLM output is parsed via Model.model_validate_json — malformed/
incomplete JSON surfaces as a ValidationError (caught above, in the loop, and written to the trace)
rather than leaking garbage (§3).

The "task → model" routing is decided by the gateway itself via model_map[task] ("plan"/"reflect"/"synth") —
the steps do not know provider names (CLAUDE.md §2, One LLM gateway).
"""
from __future__ import annotations

from collections.abc import Iterator

from rag.contracts import ExpandedSection, Reflection, RetrievedChunk, SearchPlan


# --- PLAN (Q6) -----------------------------------------------------------------------

DEFAULT_CORPUS_LANGUAGE = "English"


def _plan_system(corpus_language: str) -> str:
    # Cross-lingual (§5): the corpus is in `corpus_language`, but the question may be in any language.
    # BM25 is LEXICAL full-text → queries not in the corpus language yield 0 hits; vector degrades
    # cross-lingually. So the search queries MUST be in the corpus language (we translate the concepts).
    return (
        "You are a retrieval planner for a closed-corpus RAG system. "
        "Decompose the user's question into 2-5 atomic sub-questions, then derive search queries: "
        "`vector_queries` are expanded natural-language reformulations (for semantic search) and "
        "`bm25_queries` are short keyword queries (for full-text search). "
        f"The document corpus is written in {corpus_language}. Produce ALL `vector_queries` and "
        f"`bm25_queries` in {corpus_language}, translating the user's terms as needed, regardless of "
        "the language the question is written in (`sub_questions` may stay in the question's language). "
        "Respond with ONLY a JSON object with keys `sub_questions`, `vector_queries`, `bm25_queries`, "
        "each a list of strings. No prose, no markdown, no code fences."
    )


def plan(question: str, *, gateway, corpus_language: str = DEFAULT_CORPUS_LANGUAGE) -> SearchPlan:
    """Question → SearchPlan. An empty/whitespace question is rejected BEFORE calling the gateway.

    `corpus_language` is the corpus language; search queries are translated into it (cross-lingual search)."""
    q = (question or "").strip()
    if not q:
        raise ValueError("PLAN requires a non-empty question")
    messages = [
        {"role": "system", "content": _plan_system(corpus_language)},
        {"role": "user", "content": f"Question: {q}"},
    ]
    raw = gateway.completion(task="plan", messages=messages)
    return SearchPlan.model_validate_json(raw)


# --- REFLECT (Q7, source-aware) ------------------------------------------------------

def _reflect_system(corpus_language: str) -> str:
    return (
        "You are a retrieval critic for a closed-corpus RAG system (there is no web fallback). "
        "Given the question, its sub-questions, and the evidence retrieved so far (grouped by source "
        "document), judge whether the evidence is sufficient to answer every sub-question. Report which "
        "sub-questions are answered, which gaps remain, and per-document coverage. If gaps remain, "
        "propose NEW search queries (`new_vector_queries`, `new_bm25_queries`) that would close them; "
        "if nothing in the corpus could close a gap, leave the new queries empty. "
        f"The corpus is written in {corpus_language}; produce the new queries in {corpus_language} "
        "(translate as needed), since they feed lexical and semantic search over that corpus. "
        "Respond with ONLY a JSON object with keys: is_sufficient (bool), answered (list of strings), "
        "gaps (list of strings), coverage_by_document (list of {document, covered, missing}), "
        "new_vector_queries (list of strings), new_bm25_queries (list of strings). "
        "No prose, no markdown, no code fences."
    )


def _doc_label(source: str) -> str:
    """Document label from source ('{title} › {path}') — the part before ' › ' (for coverage_by_document)."""
    return source.split(" › ")[0] if source else ""


def _group_by_document(chunks: list[RetrievedChunk]) -> list[tuple[str, list[RetrievedChunk]]]:
    """Groups chunks by document_id preserving order of appearance; label = the document."""
    groups: dict[int, list[RetrievedChunk]] = {}
    label: dict[int, str] = {}
    order: list[int] = []
    for c in chunks:
        if c.document_id not in groups:
            groups[c.document_id] = []
            label[c.document_id] = _doc_label(c.source)
            order.append(c.document_id)
        groups[c.document_id].append(c)
    return [(label[d], groups[d]) for d in order]


def _reflect_user(question: str, plan: SearchPlan, chunks: list[RetrievedChunk]) -> str:
    lines = [f"Question: {question}", "", "Sub-questions:"]
    lines += [f"- {sq}" for sq in plan.sub_questions]
    lines += ["", "Evidence retrieved so far (grouped by source document):"]
    grouped = _group_by_document(chunks)
    if not grouped:
        lines.append("(none)")
    for source, group in grouped:
        lines.append(f"## {source}")
        lines += [f"- [{c.chunk_id}] {c.content}" for c in group]
    return "\n".join(lines)


def reflect(
    question: str,
    plan: SearchPlan,
    chunks: list[RetrievedChunk],
    *,
    gateway,
    relevance_threshold: float = 0.0,
    corpus_language: str = DEFAULT_CORPUS_LANGUAGE,
) -> Reflection:
    """Chunks + plan → Reflection (source-aware). The CRAG relevance threshold drops weak chunks
    from the material BEFORE the critic (score < threshold → not included in the prompt; 0.0 → we drop nothing).
    `corpus_language` — the newly proposed queries are produced in it (they go back into the searchers)."""
    evidence = [c for c in chunks if c.score >= relevance_threshold]
    messages = [
        {"role": "system", "content": _reflect_system(corpus_language)},
        {"role": "user", "content": _reflect_user((question or "").strip(), plan, evidence)},
    ]
    raw = gateway.completion(task="reflect", messages=messages)
    return Reflection.model_validate_json(raw)


# --- SYNTH (Q8, stream over the blocks' full_text) -----------------------------------

_SYNTH_SYSTEM = (
    "You are a technical writer answering strictly from the provided sources (a closed corpus). "
    "Use ONLY the material below; do not invent facts. Cite the sources you rely on inline using "
    "their bracketed label, e.g. [source]. If the material does not answer the question, say so plainly. "
    "Write the answer in the SAME language as the question (the sources may be in another language — "
    "translate the substance into the question's language); keep the bracketed [source] labels verbatim."
)


def _synth_user(question: str, blocks: list[ExpandedSection]) -> str:
    lines = [f"Question: {question}", "", "Sources:"]
    for b in blocks:
        lines.append(f"[{b.source}]")     # cite as [source]
        lines.append(b.full_text)         # synthesize over the chapter's full_text, NOT over chunk trimmings
        lines.append("")
    lines.append("Write the answer, citing the sources you use inline as [source].")
    return "\n".join(lines)


def synth_stream(question: str, blocks: list[ExpandedSection], *, gateway) -> Iterator[str]:
    """Question + blocks (ExpandedSection.full_text) → a stream of answer tokens (for the SSE stream)."""
    messages = [
        {"role": "system", "content": _SYNTH_SYSTEM},
        {"role": "user", "content": _synth_user((question or "").strip(), blocks)},
    ]
    return gateway.completion_stream(task="synth", messages=messages)


def synth(question: str, blocks: list[ExpandedSection], *, gateway) -> str:
    """Non-streaming facade: joins the token stream into a finished string (for CLI/run())."""
    return "".join(synth_stream(question, blocks, gateway=gateway))
