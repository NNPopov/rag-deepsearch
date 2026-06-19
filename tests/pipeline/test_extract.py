"""R10 (red): Extract filter `ingest.extract.ExtractStep` (S7).

Pins contract §4.1 Rag_pipeline_architecture.md:
  • filter step (Step contract from S6): `name`, `params()`, `artifacts`, `run(ctx)`;
  • extracts flat text from the source PDF and writes it to `01_extract/full_text.txt`
    (markers are NOT stripped — Structure does that);
  • the extraction method is an EXPLICIT parameter (default pdfminer, which produced a working full_text.txt),
    not a silent fallback chain; an empty result → error (not a silent empty file);
  • `params()` includes the source and method → changing either invalidates the step (S6 cascade).

The backend function is injected (like litellm in the gateway) → a pure unit with no real PDF/network.
RED before S7: the ingest.extract package does not exist yet.
"""
import pytest


def _step(**kw):
    from ingest.extract import ExtractStep

    kw.setdefault("source", "book.pdf")
    return ExtractStep(**kw)


def _ctx(tmp_path, guid="g1", stage="01_extract"):
    from ingest.orchestrator import StepContext
    from ingest.rundir import RunDir

    rd = RunDir(tmp_path / "runs", guid)
    return StepContext(rundir=rd, guid=guid, stage_dir=rd.stage_dir(stage))


# --- Step contract -------------------------------------------------------------
def test_conforms_to_step_protocol():
    from ingest.orchestrator import Step

    step = _step(extract_fn=lambda src: "x")
    assert isinstance(step, Step)
    assert step.name == "01_extract"
    assert list(step.artifacts) == ["01_extract/full_text.txt"]


def test_params_include_source_and_method():
    step = _step(source="DDIA.pdf", method="pdfminer", extract_fn=lambda src: "x")
    p = step.params()
    assert p["source"] == "DDIA.pdf"
    assert p["method"] == "pdfminer"


def test_default_method_is_pdfminer():
    step = _step(extract_fn=lambda src: "x")
    assert step.params()["method"] == "pdfminer"


# --- extraction ----------------------------------------------------------------
def test_run_writes_full_text_and_returns_path(tmp_path):
    step = _step(extract_fn=lambda src: "Chapter 1\nhello")
    ctx = _ctx(tmp_path)
    artifact = step.run(ctx)
    out = ctx.stage_dir / "full_text.txt"
    assert out.read_text(encoding="utf-8") == "Chapter 1\nhello"
    assert artifact == str(out)


def test_run_passes_configured_source_to_backend(tmp_path):
    seen = {}

    def fake(src):
        seen["src"] = src
        return "text"

    _step(source="path/to/DDIA.pdf", extract_fn=fake).run(_ctx(tmp_path))
    assert seen["src"] == "path/to/DDIA.pdf"


def test_markers_not_stripped(tmp_path):
    raw = "CHAPTER 1\nFoundations\n\fpage break\nChapter Title  | 12"
    step = _step(extract_fn=lambda src: raw)
    ctx = _ctx(tmp_path)
    step.run(ctx)
    assert (ctx.stage_dir / "full_text.txt").read_text(encoding="utf-8") == raw


def test_empty_extraction_raises(tmp_path):
    for empty in ("", "   \n\t  ", None):
        step = _step(extract_fn=lambda src, _e=empty: _e)
        with pytest.raises(Exception):
            step.run(_ctx(tmp_path, stage="01_extract_%s" % id(empty)))


def test_backends_registry_has_known_methods():
    from ingest.extract import BACKENDS

    assert {"pdfminer", "pypdf2"} <= set(BACKENDS)
    assert all(callable(fn) for fn in BACKENDS.values())


def test_unknown_method_rejected(tmp_path):
    step = _step(method="nope")  # no injection → resolve from the registry
    with pytest.raises(Exception):
        step.run(_ctx(tmp_path))


# --- integration with S6 ----------------------------------------------------------
def test_works_as_orchestrator_step(tmp_path):
    from ingest.orchestrator import Orchestrator
    from ingest.rundir import RunDir

    step = _step(extract_fn=lambda src: "full text body")
    Orchestrator([step], base=tmp_path / "runs").run("g1")

    root = tmp_path / "runs" / "g1"
    assert (root / "01_extract" / "full_text.txt").read_text(encoding="utf-8") == "full text body"
    state = RunDir(tmp_path / "runs", "g1").load_state()["steps"]["01_extract"]
    assert state["status"] == "done"
    assert state["artifacts"] == ["01_extract/full_text.txt"]
