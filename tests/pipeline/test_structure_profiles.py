"""S8-hardening (red): structure profiles in `ingest.structure` (design note §4.2.1, steps 0→1).

Contract §4.2/§4.2.1 Rag_pipeline_architecture.md: `StructureStep` is a mechanism with TWO injectable
detectors; domain knowledge about book formats lives in the `BOOK_PROFILES` registry:
  • profile = {`chapter`: running-header regex, `subsections`: subsection detection mode};
  • `ddia`    = `Chapter N: Title` + subsections from split footers (`footers`);
  • `manning` = `chapter N  Title` (lowercase, two+ spaces) + subsections from NUMBERED body
    headings `N.M[.K]  Title` (`numbered`) — Acing the System Design Interview.

Step 1: Manning RECOVERS subsections (not flat chapters). In the Acing body, numbered
headings sit as clean one-liners (`1.4.2  Scaling with GeoDNS`, `2.1  Clarify…`) → cut
points; chapter = depth-1 big-block (small2big), subsections = depth-2 leaves (like DDIA, path `N.k`
by order of appearance, heading = the real numbered heading). Known limitation (§4.2.1):
an `N.M` part where pdfminer detached the number from the heading is not caught (the heading fragment merges into
the neighboring text); the chapter opener before the first running header leaks.

The MANNING_TEXT fixture is grounded on real pdfminer Acing (runs/acing-q12/01_extract).
Pure unit, no PDF/network. RED before step 1.
"""
from __future__ import annotations

import pytest

from ingest.contracts import Section

# --- fixture: real pdfminer Acing format (running header `chapter N  Title` + numbered headings) ---
MANNING_TEXT = """\
1A walkthrough of

system design concepts

This chapter covers a system design interview discussion.

chapter 1  A walkthrough of system design concepts

A discussion about tradeoffs
The following factors attest to the importance of system design interviews.

chapter 1  A walkthrough of system design concepts

1.3  Overview of this book
This book offers a structured approach to preparing for system design interviews.

chapter 1  A walkthrough of system design concepts

1.4.2  Scaling with GeoDNS
GeoDNS routes users to the nearest data center to reduce latency.

chapter 2  A typical system design interview flow

2.1  Clarify requirements and discuss tradeoffs
Clarifying the requirements of the system is the first step in the interview.

chapter 2  A typical system design interview flow

2.2  Draft the API specification
Draft the API specification before designing the data model.
"""


# --- profile registry -------------------------------------------------------------------------
def test_book_profiles_registry_has_ddia_and_manning():
    from ingest.structure import BOOK_PROFILES, resolve_profile

    assert {"ddia", "manning"} <= set(BOOK_PROFILES)
    ddia = resolve_profile("ddia")
    assert ddia["subsections"] == "footers"
    manning = resolve_profile("manning")
    assert manning["subsections"] == "numbered" and manning["chapter"]
    with pytest.raises(ValueError):
        resolve_profile("no-such-profile")


def test_ddia_profile_is_default():
    """StructureStep default is the ddia profile (footers), backward compatible with S8."""
    from ingest.structure import BOOK_PROFILES, StructureStep

    step = StructureStep(external_id="x")
    p = step.params()
    assert p["chapter_pattern"] == BOOK_PROFILES["ddia"]["chapter"]
    assert p["subsection_mode"] == "footers"


# --- manning: chapters by running header + subsections by numbered body headings ---------------
@pytest.fixture()
def manning_step():
    from ingest.structure import StructureStep, resolve_profile

    p = resolve_profile("manning")
    return StructureStep(
        external_id="acing", chapter_pattern=p["chapter"], subsection_mode=p["subsections"]
    )


def test_manning_detects_chapters(manning_step):
    chapters = [s for s in manning_step.parse(MANNING_TEXT) if s.depth == 1]
    assert [c.path for c in chapters] == ["1", "2"]
    assert [c.heading for c in chapters] == [
        "A walkthrough of system design concepts",
        "A typical system design interview flow",
    ]
    assert all(c.parent_path is None for c in chapters)


def test_manning_recovers_numbered_subsections(manning_step):
    """Chapters are NOT flat: numbered body headings → depth-2 leaves (step 1)."""
    sections = manning_step.parse(MANNING_TEXT)
    subs = [s for s in sections if s.depth == 2]
    headings = [s.heading for s in subs]
    assert "1.3  Overview of this book" in headings
    assert "1.4.2  Scaling with GeoDNS" in headings
    assert "2.1  Clarify requirements and discuss tradeoffs" in headings
    assert "2.2  Draft the API specification" in headings
    assert all(s.is_leaf for s in subs)            # subsections are leaves (go to Chunk)


def test_manning_chapter_is_big_block_not_leaf(manning_step):
    by_path = {s.path: s for s in manning_step.parse(MANNING_TEXT)}
    assert by_path["1"].is_leaf is False           # a chapter with subsections — a small2big big-block
    # the chapter text (big-block) contains the prose of its subsections
    assert "GeoDNS routes users" in by_path["1"].text


def test_manning_subsection_parent_and_text_sliced(manning_step):
    sections = manning_step.parse(MANNING_TEXT)
    by_heading = {s.heading: s for s in sections}
    geodns = by_heading["1.4.2  Scaling with GeoDNS"]
    assert geodns.depth == 2 and geodns.parent_path == "1"   # ← small2big: ancestor = chapter 1
    assert "GeoDNS routes users" in geodns.text
    assert "Overview of this book" not in geodns.text        # its own subsection slice
    # the running header and number line don't leak into the text
    assert "chapter 1" not in geodns.text


def test_manning_sections_are_valid(manning_step):
    sections = manning_step.parse(MANNING_TEXT)
    assert sections
    for s in sections:
        assert isinstance(s, Section)
        assert s.external_id == "acing"
        assert s.token_count > 0


def test_manning_profile_fails_on_ddia_default():
    """With the default (ddia) running header, Manning text yields no chapters → StructureError."""
    from ingest.structure import StructureError, StructureStep

    with pytest.raises(StructureError):
        StructureStep(external_id="acing").parse(MANNING_TEXT)
