"""R6 (red): Structure filter `ingest.structure.StructureStep` (S8).

Pins the §4.2 Rag_pipeline_architecture.md contract — ONE filter cuts flat text
into a section tree (chapter = depth 1, subchapter = depth 2) and writes `02_structure/sections.jsonl`
(one valid `Section` per line — the Structure→Chunk pipe from §3 / contracts.py).

IMPORTANT — boundary signals are validated on REAL pdfminer.six output (Designing_Data.pdf, pp. 28–46),
not on the idealized format of the old doc:
  • chapter  = running-header line `Chapter N: <Title>` (repeated on every page of the
    chapter) → reliably yields BOTH the number AND the chapter heading; the chapter boundary = where the number changes;
  • subchapters = BOTTOM footers `<Section Name> | <pagenum>`, which pdfminer splits into THREE
    non-empty lines (name / `|` / number). Recovering them yields a trustworthy list of section names;
    each name appears in the body as a clean heading line → the cut point.
docling is NOT used (see §7 CLAUDE.md). Pure unit: fixture in real pdfminer format,
no PDF/network. RED before S8: the ingest.structure module does not exist yet.
"""
from __future__ import annotations

import json

import pytest

import db
from ingest.contracts import Section
from ingest.orchestrator import Orchestrator
from ingest.rundir import RunDir

# --- fixture: format like real pdfminer (running headers + split footers) -------------------
# Blank lines between footer fragments and a repeated running header — as in the live output.
DDIA_TEXT = """\
Chapter 1: Trade-Offs in Data Systems Architecture

Trade-Offs in Data Systems Architecture

Data is at the center of many challenges in modern system design.
This chapter introduces the trade-offs that recur throughout the book.

Operational Versus Analytical Systems

If you are building data systems you encounter two broad categories of
workloads: operational and analytical.

Operational Versus Analytical Systems

|

5

Operational systems handle low-latency reads and writes by key.

Cloud Versus Self-Hosting

A major decision is whether to run your own infrastructure or rent it.

Cloud Versus Self-Hosting

|

9

Chapter 2: Defining Nonfunctional Requirements

Defining Nonfunctional Requirements

Functional requirements describe what an application should do.

Reliability

Reliability means the system continues to work correctly even when
things go wrong.

Reliability

|

55
"""


@pytest.fixture()
def step():
    from ingest.structure import StructureStep

    return StructureStep(external_id="ddia")


# --- Step contract (S6) --------------------------------------------------------------------
def test_conforms_to_step_protocol(step):
    from ingest.orchestrator import Step

    assert isinstance(step, Step)
    assert step.name == "02_structure"
    assert "02_structure/sections.jsonl" in step.artifacts


def test_params_include_external_id_and_pattern(step):
    p = step.params()
    assert p["external_id"] == "ddia"
    assert "chapter_pattern" in p  # changing the detection pattern → step invalidation (S6 cascade)


# --- chapter detection from the running header ---------------------------------------------
def test_detects_chapters_from_running_header(step):
    sections = step.parse(DDIA_TEXT)
    chapters = [s for s in sections if s.depth == 1]
    assert [c.path for c in chapters] == ["1", "2"]
    assert [c.heading for c in chapters] == [
        "Trade-Offs in Data Systems Architecture",
        "Defining Nonfunctional Requirements",
    ]
    assert [c.ordinal for c in chapters] == [1, 2]
    assert all(c.parent_path is None for c in chapters)


# --- subchapter detection from recovered footers -------------------------------------------
def test_detects_subchapters_from_footers(step):
    sections = step.parse(DDIA_TEXT)
    subs = [s for s in sections if s.depth == 2 and s.ordinal > 0]
    headings = [s.heading for s in subs]
    assert "Operational Versus Analytical Systems" in headings
    assert "Cloud Versus Self-Hosting" in headings
    assert "Reliability" in headings


def test_subchapter_paths_and_parent(step):
    sections = step.parse(DDIA_TEXT)
    by_heading = {s.heading: s for s in sections}
    op = by_heading["Operational Versus Analytical Systems"]
    assert op.depth == 2
    assert op.parent_path == "1"
    assert op.path == "1.1"  # first subchapter of chapter 1
    cloud = by_heading["Cloud Versus Self-Hosting"]
    assert cloud.path == "1.2"
    rel = by_heading["Reliability"]
    assert rel.parent_path == "2" and rel.path == "2.1"


# --- leaves / big-block (small2big) --------------------------------------------------------
def test_leaf_flags(step):
    sections = step.parse(DDIA_TEXT)
    by_path = {s.path: s for s in sections}
    assert by_path["1"].is_leaf is False  # a chapter with subchapters is not a leaf
    assert by_path["1.1"].is_leaf is True  # a subchapter is a leaf (goes to Chunk)


def test_chapter_text_is_big_block_and_subchapter_text_is_slice(step):
    sections = step.parse(DDIA_TEXT)
    by_path = {s.path: s for s in sections}
    chapter = by_path["1"]
    op = by_path["1.1"]
    # the chapter big-block contains the prose of all its subchapters
    assert "low-latency reads and writes" in chapter.text
    assert "run your own infrastructure" in chapter.text
    # a subchapter's text is its own slice, without the neighboring subchapter's prose
    assert "low-latency reads and writes" in op.text
    assert "run your own infrastructure" not in op.text


def test_section_text_is_clean_of_markers(step):
    """Running headers and footer fragments (`Name`/`|`/number) don't leak into the section text."""
    sections = step.parse(DDIA_TEXT)
    op = next(s for s in sections if s.heading == "Operational Versus Analytical Systems")
    assert "Chapter 1:" not in op.text
    assert "|" not in op.text
    assert "\n5" not in op.text and op.text.strip() != "5"


def test_page_range_best_effort_from_footer(step):
    sections = step.parse(DDIA_TEXT)
    op = next(s for s in sections if s.heading == "Operational Versus Analytical Systems")
    assert op.page_range == (5, 5)


# --- all lines are valid Sections, tokens counted -----------------------------------------
def test_all_nodes_are_valid_sections(step):
    sections = step.parse(DDIA_TEXT)
    assert sections, "should have sections"
    for s in sections:
        assert isinstance(s, Section)
        assert s.external_id == "ddia"
        assert s.token_count > 0


# --- empty / chapterless input → error, not silent emptiness -------------------------------
def test_empty_or_chapterless_text_raises(step):
    from ingest.structure import StructureError

    with pytest.raises(StructureError):
        step.parse("just some prose with no chapter running headers at all\n")


# --- works as an orchestrator step: reads 01_extract, writes sections.jsonl ----------------
def test_runs_as_orchestrator_step(tmp_path):
    from ingest.structure import StructureStep

    base = tmp_path / "runs"
    guid = "g-struct"
    rd = RunDir(base, guid)
    extract_dir = rd.stage_dir("01_extract")
    (extract_dir / "full_text.txt").write_text(DDIA_TEXT, encoding="utf-8")

    Orchestrator([StructureStep(external_id="ddia")], base=base).run(guid)

    out = rd.path("02_structure", "sections.jsonl")
    assert out.exists()
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "sections.jsonl must not be empty"
    # each line is a valid Section (the Structure→Chunk pipe)
    parsed = [Section.model_validate_json(ln) for ln in lines]
    assert any(s.depth == 1 for s in parsed)
    assert any(s.depth == 2 and s.is_leaf for s in parsed)
    # the contract vector length is irrelevant here, but db must import (the shared layer)
    assert db.DIMENSION == 1536
