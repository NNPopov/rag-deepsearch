"""QR10 (pure helpers) — `rag.loop._cited_sources` citation matching.

The DeepSearch run()/stream() orchestration tests moved to test_rag_loop_async.py (ADR-0002: the
loop is async). What stays here are the SYNC pure-helper unit tests: `_cited_sources` does no I/O
(ADR-0002 §3 — pure-CPU functions are not colored async), so it is tested directly, no event loop.

Basis: Rag_query_architecture.md §3 (citation accuracy: robust label matching).
"""
from __future__ import annotations


# --- citation accuracy: robust label matching --------------------------------------

def test_cited_sources_matches_canonical_label():
    """The model reproduced the label verbatim [title › path] → exact match."""
    from rag.loop import _cited_sources

    answer = "We rely on [Designing Data-Intensive Applications › 3] here."
    blocks = [
        "Designing Data-Intensive Applications › 3",
        "Acing the System Design Interview › 4",
    ]
    assert _cited_sources(answer, blocks) == ["Designing Data-Intensive Applications › 3"]


def test_cited_sources_robust_to_comma_and_deep_path():
    """flash bug: comma instead of › and a deep number (1.4.6, 4.8). We match book+chapter."""
    from rag.loop import _cited_sources

    answer = (
        "CQRS [Acing the System Design Interview, 1.4.6]. "
        "CDC [Designing Data-Intensive Applications, 12]. "
        "Cache [Acing the System Design Interview, 4.8]."
    )
    blocks = [
        "Acing the System Design Interview › 1",
        "Designing Data-Intensive Applications › 12",
        "Acing the System Design Interview › 4",
        "Designing Data-Intensive Applications › 8",   # not mentioned → must not be included
    ]
    assert _cited_sources(answer, blocks) == [
        "Acing the System Design Interview › 1",
        "Designing Data-Intensive Applications › 12",
        "Acing the System Design Interview › 4",
    ]


def test_cited_sources_ignores_chapter_mentions_outside_brackets():
    """"chapter 8" in prose (outside brackets) is not a citation; we count only what is in [...]."""
    from rag.loop import _cited_sources

    answer = "As discussed in chapter 8, indexes help. We cite [DDIA › 3]."
    blocks = ["DDIA › 3", "DDIA › 8"]
    assert _cited_sources(answer, blocks) == ["DDIA › 3"]


def test_cited_sources_empty_when_no_brackets():
    from rag.loop import _cited_sources

    assert _cited_sources("no citations here", ["DDIA › 3"]) == []
