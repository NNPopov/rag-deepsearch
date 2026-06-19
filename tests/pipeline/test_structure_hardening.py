"""S8 hardening (red): chapter opener + back-matter in `ingest.structure` (§4.2 / §4.2.1).

Two §4.2 detector flaws found on LIVE DDIA (runs/ddia-s14) are pinned here as contract:

1. **The preamble swallows the first subchapter.** The chapter's opening page (caps marker `CHAPTER N`,
   heading, intro prose, the FIRST subheading) comes in pdfminer BEFORE the running header
   `Chapter N: Title` first repeats. The old detector cut the chapter on the running header → the whole opener
   leaked into the previous chapter (cur=None → discarded), and the current chapter's first subchapter was left
   without a clean heading line in the body → it wasn't cut → its prose collapsed into the `N.0` preamble.
   Fix: the true chapter boundary is the caps opener `CHAPTER N` (reliable: exactly one per chapter, before
   the running header); the heading is taken from the running header. The opener catches both the intro (→`N.0`) and the first
   subheading (→`N.1`).

2. **Back-matter (Glossary/Index) in the tree.** `Glossary`/`Index` footers after the LAST chapter
   attach to it (cur_chapter doesn't change) → land as subchapters and in the big-block (Index = a sheet of
   "term, page" — pure noise for retrieval). Fix: the last chapter's body is truncated at the first
   back-matter heading (Glossary/Index/Bibliography/Colophon) — both the heading itself and
   everything after are dropped.

Signals grounded on real pdfminer output (caps opener `CHAPTER 6`+`Replication` at L10500,
first subheading `Single-Leader Replication` at L10579 — BEFORE the first `Chapter 6:` running header at
L10588; back-matter `Glossary`/`Index` as footers after ch.14). Pure unit, no PDF/network. RED
before S8-hardening.
"""
from __future__ import annotations

import pytest

from ingest.contracts import Section

# --- fixture: real pdfminer format — caps opener BEFORE the running header, split footers,
#     back-matter (Glossary/Index) after the last chapter. ----------------------------------------
# Ch.1: opener `CHAPTER 1`+title, intro (→1.0), first subheading `Operational…` is SEEN as a clean
# line ONLY in the opener (after the running header — just a footer fragment), then `Cloud…`. Without the fix
# `Operational…` collapses into the preamble. Ch.2 (the last) — then back-matter Glossary/Index.
DDIA_OPENER_TEXT = """\
CHAPTER 1

Trade-Offs in Data Systems Architecture

This chapter introduces the trade-offs that recur throughout the book.

Operational Versus Analytical Systems

Operational systems handle low-latency reads by key.

1

Chapter 1: Trade-Offs in Data Systems Architecture

Operational systems also handle writes durably across replicas.

Operational Versus Analytical Systems

|

5

Cloud Versus Self-Hosting

A major decision is whether to run your own infrastructure or rent it.

Cloud Versus Self-Hosting

|

9

CHAPTER 2

Defining Nonfunctional Requirements

Functional requirements describe what an application should do.

Reliability

Reliability means the system keeps working correctly when things go wrong.

Chapter 2: Defining Nonfunctional Requirements

Reliability

|

55

Glossary

asynchronous Not waiting for an operation to complete before proceeding.

Glossary

|

651

Index

replication 197, sharding 199, transactions 221

Index

|

659
"""


@pytest.fixture()
def step():
    from ingest.structure import StructureStep

    return StructureStep(external_id="ddia")


# --- opener boundary: chapters detected by the caps marker `CHAPTER N`, heading from the running header ----
def test_detects_chapters_from_caps_opener(step):
    sections = step.parse(DDIA_OPENER_TEXT)
    chapters = [s for s in sections if s.depth == 1]
    assert [c.path for c in chapters] == ["1", "2"]
    assert [c.heading for c in chapters] == [
        "Trade-Offs in Data Systems Architecture",
        "Defining Nonfunctional Requirements",
    ]


# --- FIX 1: the first subchapter is NOT swallowed by the preamble -----------------------------
def test_first_subsection_not_swallowed_into_preamble(step):
    sections = step.parse(DDIA_OPENER_TEXT)
    headings = {s.heading for s in sections if s.depth == 2}
    # both subchapters of chapter 1 are present — previously the first ("Operational…") sank into the preamble
    assert "Operational Versus Analytical Systems" in headings
    assert "Cloud Versus Self-Hosting" in headings
    by_heading = {s.heading: s for s in sections}
    op = by_heading["Operational Versus Analytical Systems"]
    cloud = by_heading["Cloud Versus Self-Hosting"]
    assert op.parent_path == "1" and op.path == "1.1"   # the FIRST subchapter
    assert cloud.path == "1.2"
    assert "low-latency reads" in op.text               # its prose — in its leaf


def test_chapter_intro_is_preamble_not_lost(step):
    """The opener's intro prose (before the first subheading) is kept as preamble N.0, not lost."""
    sections = step.parse(DDIA_OPENER_TEXT)
    by_path = {s.path: s for s in sections}
    assert "1.0" in by_path                                   # the preamble leaf exists
    pre = by_path["1.0"]
    assert "introduces the trade-offs" in pre.text            # intro preserved
    assert "low-latency reads" not in pre.text                # but NOT the first subchapter's prose


def test_preamble_excludes_first_subsection_text(step):
    """The chapter big-block contains the subchapters' prose, but the preamble leaf doesn't duplicate the first subchapter."""
    sections = step.parse(DDIA_OPENER_TEXT)
    by_path = {s.path: s for s in sections}
    assert "low-latency reads" in by_path["1"].text           # big-block (small2big) — the whole chapter
    assert "run your own infrastructure" in by_path["1"].text


# --- FIX 2: back-matter (Glossary/Index) dropped from the tree and from the big-block ---------
def test_back_matter_dropped_from_tree(step):
    sections = step.parse(DDIA_OPENER_TEXT)
    headings = {s.heading for s in sections}
    assert "Glossary" not in headings
    assert "Index" not in headings


def test_back_matter_text_not_in_last_chapter_block(step):
    sections = step.parse(DDIA_OPENER_TEXT)
    by_path = {s.path: s for s in sections}
    ch2 = by_path["2"]
    assert "asynchronous Not waiting" not in ch2.text         # glossary prose — not in the chapter
    assert "replication 197" not in ch2.text                  # the index sheet — not in the chapter
    # the legitimate subchapter of the last chapter survived
    assert "Reliability" in {s.heading for s in sections if s.depth == 2}


# --- all nodes are valid Sections -----------------------------------------------------------
def test_hardened_sections_are_valid(step):
    sections = step.parse(DDIA_OPENER_TEXT)
    assert sections
    for s in sections:
        assert isinstance(s, Section)
        assert s.external_id == "ddia"
        assert s.token_count > 0


# --- restart contract: changing the opener/back-matter invalidates the step --------------------
def test_params_include_opener_and_back_matter(step):
    p = step.params()
    assert "opener_pattern" in p          # changing opener detection → re-cut cascade (S6)
    assert "back_matter" in p


# --- backward compatibility: without caps markers the detector falls back to the running header (old path) ------
def test_falls_back_to_running_header_when_no_opener_markers(step):
    """Text without `CHAPTER N` (the old fixture) is still cut on the running header `Chapter N:`."""
    no_opener = (
        "Chapter 1: Alpha\n\nAlpha\n\nIntro prose for alpha chapter.\n\n"
        "Widgets\n\nWidgets are useful for many tasks.\n\nWidgets\n\n|\n\n5\n"
    )
    sections = step.parse(no_opener)
    chapters = [s for s in sections if s.depth == 1]
    assert [c.path for c in chapters] == ["1"]
    assert chapters[0].heading == "Alpha"
    assert "Widgets" in {s.heading for s in sections if s.depth == 2}


def test_back_matter_only_truncates_final_chapter(step):
    """Back-matter is truncated only at the LAST chapter — the word 'Index' as a heading in the middle is left alone
    (in DDIA back-matter is book-level anyway; this guards against false truncation of middle chapters)."""
    sections = step.parse(DDIA_OPENER_TEXT)
    # chapter 1 (not the last) fully preserved — both its subchapters are in place
    ch1_subs = [s for s in sections if s.depth == 2 and s.parent_path == "1"]
    assert {"Operational Versus Analytical Systems", "Cloud Versus Self-Hosting"} <= {
        s.heading for s in ch1_subs
    }
