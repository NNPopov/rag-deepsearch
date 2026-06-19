"""S8 (red): the `progit` profile in `ingest.structure` — Pro Git chapters by form-feed + curated set.

Contract §4.2.1 Rag_pipeline_architecture.md: Pro Git (Asciidoctor PDF) gives NEITHER a running
header `Chapter N: Title` (ddia) NOR numbered body headings `N.M  Title` (manning).
The reliable (and only) chapter-boundary signal in pdfminer output is a line starting with
form-feed `\\x0c` (a new page) whose text EXACTLY equals a chapter title from the profile's
curated set. The form-feed check resolves a collision: a chapter title often appears in prose as a
cross-reference ("as covered in Git Basics") and even as a section heading in another chapter
(git-svn "Getting Started" in the "Git and Other Systems" chapter) — but those come WITHOUT a form-feed.

Pro Git section headings are bare title-case lines without markers (no number, no footers,
no form-feed) → they can't be reliably distinguished from unformatted prose. So v1 is CHAPTER-LEVEL
structure (`subsections="none"`): a chapter = one leaf, the chunker cuts the rest. TOC-driven subsections —
§4.2.1 step 2 (later). The LAST chapter's body is truncated at the first form-feed back-matter
(`Appendix …` / `Index`), otherwise appendices and the index leak into "Git Internals".

The PROGIT_TEXT fixture is grounded on real pdfminer progit_en.pdf (work/progit/en_full.txt):
form-feed chapter openers (`\\fGit Basics`), single-number page numbers, bare-line sections,
the "Getting Started" collision. Pure unit, no PDF/network. RED before the profile is implemented.
"""
from __future__ import annotations

import pytest

from ingest.contracts import Section

# `\f` = form feed (0x0c). pdfminer places it as the first character of a line at the start of a new page;
# for a chapter opener it is the first (and only) text line of such a page.
PROGIT_TEXT = (
    "Let’s get started.\n"                         # front matter before chapter 1 — ignore
    "\n9\n\n"                                            # page number as a single number
    "\fGetting Started\n"                                # chapter 1: form-feed + exact title
    "This chapter will be about getting started with Git.\n"
    "\nAbout Version Control\n"                          # section — bare line, NOT cut in v1
    "What is version control, and why should you care?\n"
    "\n10\n\n"
    "\fGit Basics\n"                                     # chapter 2
    "If you can read only one chapter to get going, this is it.\n"
    "We covered this briefly as you saw in Getting Started.\n"  # cross-reference — NOT a chapter
    "\nGetting Started\n"                                # collision: git-svn section WITHOUT form-feed → NOT a chapter
    "Now that you have a Subversion repository you can clone it.\n"
    "\nGit Internals\n"                                  # reference line WITHOUT form-feed → NOT a chapter
    "\f Git Internals\n"                                 # chapter 3 (form-feed; a leading space is allowed)
    "Git is fundamentally a content-addressable filesystem.\n"
    "\nPlumbing and Porcelain\n"
    "We will cover the plumbing commands here.\n"
    "\fAppendix A: Git in Other Environments\n"          # back-matter → truncate the last chapter's body
    "Visual tools and IDE integrations live here.\n"
    "\fIndex\n"
    "git, 1\n"
)


def test_progit_in_registry():
    from ingest.structure import BOOK_PROFILES, resolve_profile

    assert "progit" in BOOK_PROFILES
    p = resolve_profile("progit")
    assert p["subsections"] == "none" and p["chapter"]


@pytest.fixture()
def progit_step():
    from ingest.structure import StructureStep, resolve_profile

    p = resolve_profile("progit")
    return StructureStep(
        external_id="progit", chapter_pattern=p["chapter"], subsection_mode=p["subsections"]
    )


def test_progit_detects_chapters_by_form_feed(progit_step):
    chapters = [s for s in progit_step.parse(PROGIT_TEXT) if s.depth == 1]
    assert [c.heading for c in chapters] == ["Getting Started", "Git Basics", "Git Internals"]
    assert [c.path for c in chapters] == ["1", "2", "3"]
    assert all(c.parent_path is None for c in chapters)
    assert [c.ordinal for c in chapters] == [1, 2, 3]


def test_progit_chapters_are_leaves_no_subsections(progit_step):
    sections = progit_step.parse(PROGIT_TEXT)
    assert sections
    assert all(s.depth == 1 for s in sections)       # v1: chapters only, no depth-2
    assert all(s.is_leaf for s in sections)          # chapter-leaf → goes to Chunk


def test_progit_section_headings_stay_in_chapter_body(progit_step):
    by_h = {s.heading: s for s in progit_step.parse(PROGIT_TEXT)}
    # a bare Pro Git section does NOT become a separate section — it stays in its chapter's body
    assert "About Version Control" in by_h["Getting Started"].text
    assert "What is version control" in by_h["Getting Started"].text


def test_progit_cross_reference_is_not_a_chapter(progit_step):
    sections = progit_step.parse(PROGIT_TEXT)
    chapters = [s for s in sections if s.depth == 1]
    # "Getting Started" as a reference/section without a form-feed does not spawn a second chapter
    assert [c.heading for c in chapters].count("Getting Started") == 1
    by_h = {s.heading: s for s in sections}
    assert "as you saw in Getting Started" in by_h["Git Basics"].text


def test_progit_strips_back_matter_from_last_chapter(progit_step):
    by_h = {s.heading: s for s in progit_step.parse(PROGIT_TEXT)}
    last = by_h["Git Internals"]
    assert "content-addressable" in last.text        # its own body in place
    assert "Appendix A" not in last.text             # appendix truncated
    assert "Index" not in last.text
    assert "git, 1" not in last.text


def test_progit_page_numbers_and_form_feed_not_in_text(progit_step):
    for s in progit_step.parse(PROGIT_TEXT):
        assert "\f" not in s.text and "\x0c" not in s.text   # the page marker doesn't leak into the text
    # single page numbers don't pollute the chapter body
    gs = {s.heading: s for s in progit_step.parse(PROGIT_TEXT)}["Getting Started"]
    assert "\n10\n" not in f"\n{gs.text}\n"


def test_progit_sections_valid(progit_step):
    sections = progit_step.parse(PROGIT_TEXT)
    assert sections
    for s in sections:
        assert isinstance(s, Section)
        assert s.external_id == "progit"
        assert s.token_count > 0


def test_progit_profile_fails_on_text_without_chapters(progit_step):
    """Text without a single form-feed opener from the set → StructureError, not empty output."""
    from ingest.structure import StructureError

    with pytest.raises(StructureError):
        progit_step.parse("just some prose\nwith no Pro Git chapter openers at all\n")
