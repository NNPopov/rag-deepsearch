"""R-clean (S9, optional) — Clean filter: cleaning pdfminer artifacts in section text.

Hybrid (decided 2026-06-17): default — a DETERMINISTIC normalizer (`normalize_pdf_text`,
safe/idempotent, no content drift, handles big blocks of any size); LLM cleaning via the
gateway — an injectable `clean_fn`/`method='llm'` (the seam is ready, but not the default). The step sits AFTER Structure
(`02_structure/sections.jsonl` → `03_clean/sections.jsonl`), cleans ALL sections, recomputes
`token_count`. Orchestrator step contract (S6): `run(ctx) -> artifact`. Pure unit — no DB/network.
"""
from __future__ import annotations

from ingest.clean import CleanStep, normalize_pdf_text
from ingest.contracts import Section
from ingest.orchestrator import Step


# --- deterministic normalizer --------------------------------------------------------

def test_collapses_multiple_spaces():
    assert normalize_pdf_text("data    intensive   apps") == "data intensive apps"


def test_removes_space_before_punctuation():
    assert normalize_pdf_text("a word , and another .") == "a word, and another."


def test_dehyphenates_line_breaks():
    assert normalize_pdf_text("repli-\ncation is hard") == "replication is hard"


def test_collapses_blank_line_runs_but_keeps_paragraphs():
    assert normalize_pdf_text("para one\n\n\n\npara two") == "para one\n\npara two"


def test_strips_trailing_whitespace_per_line():
    assert normalize_pdf_text("line one   \nline two\t\n") == "line one\nline two"


def test_normalize_is_idempotent():
    raw = "a  messy -\ntext  ,  with junk .\n\n\n\nend   "
    once = normalize_pdf_text(raw)
    assert normalize_pdf_text(once) == once


# --- CleanStep as an orchestrator step -----------------------------------------------

def _section(text: str, *, path="1", depth=1, is_leaf=True) -> Section:
    return Section(
        external_id="ddia", path=path, depth=depth, parent_path=None, ordinal=1,
        heading="Replication", is_leaf=is_leaf, text=text, token_count=999,
    )


def test_cleanstep_is_a_step_with_stage_name_and_artifact():
    step = CleanStep()
    assert isinstance(step, Step)
    assert step.name == "03_clean"
    assert "03_clean/sections.jsonl" in step.artifacts


def test_params_carry_method_for_restart_invalidation():
    assert CleanStep().params()["method"] == "normalize"
    assert CleanStep(method="llm", gateway=object()).params()["method"] == "llm"


def test_clean_sections_cleans_text_and_recomputes_token_count():
    # inject count_fn=len → deterministic; the default clean_fn = normalize
    step = CleanStep(count_fn=len)
    [out] = step.clean_sections([_section("data    intensive ,  yes")])
    assert out.text == "data intensive, yes"
    assert out.token_count == len("data intensive, yes")    # recomputed, not inherited 999


def test_clean_sections_preserves_structural_fields():
    step = CleanStep(count_fn=len)
    src = _section("messy   text", path="1.2", depth=2, is_leaf=True)
    [out] = step.clean_sections([src])
    assert (out.path, out.depth, out.is_leaf, out.heading) == ("1.2", 2, True, "Replication")
    assert out.external_id == "ddia"


def test_clean_sections_processes_all_sections_including_big_blocks():
    step = CleanStep(count_fn=len)
    secs = [
        _section("chapter   body", path="1", depth=1, is_leaf=False),   # big block
        _section("leaf   prose", path="1.1", depth=2, is_leaf=True),    # leaf
    ]
    out = step.clean_sections(secs)
    assert [s.text for s in out] == ["chapter body", "leaf prose"]


def test_explicit_clean_fn_overrides_default():
    step = CleanStep(clean_fn=str.upper, count_fn=len)
    [out] = step.clean_sections([_section("hi")])
    assert out.text == "HI"


# --- LLM seam (method='llm' via the gateway) -----------------------------------------

class _FakeGateway:
    def __init__(self) -> None:
        self.calls: list = []

    def completion(self, *, task: str, messages: list[dict], **kw) -> str:
        self.calls.append((task, messages))
        return "cleaned by llm"


def test_llm_method_routes_through_gateway_clean_task():
    gw = _FakeGateway()
    step = CleanStep(method="llm", gateway=gw, count_fn=len)
    [out] = step.clean_sections([_section("raw  text")])
    assert out.text == "cleaned by llm"
    assert gw.calls and gw.calls[0][0] == "clean"          # task='clean' via the gateway
