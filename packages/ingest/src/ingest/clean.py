"""Clean filter (S9, OPTIONAL) — cleanup of pdfminer artifacts in section text.

Single responsibility (§5 Rag_pipeline_architecture.md): read `02_structure/sections.jsonl`,
strip extraction artifacts from EVERY section's text (extra spaces, space before punctuation,
hyphen line breaks), re-compute `token_count` and write `03_clean/sections.jsonl` — the same
`Section`s but with clean text. Sits STRICTLY AFTER Structure (chapter/footer markers already
removed — we clean only the prose, never touch structure). Implements the orchestrator step
contract (S6): `run(ctx) -> artifact`.

Hybrid (decided 2026-06-17): the default is the DETERMINISTIC normalizer (`normalize_pdf_text`):
safe, idempotent, doesn't change meaning/citations, handles big-blocks of any size. LLM cleanup via
the gateway (`method='llm'`) is an injectable seam: pricier/slower and risks rewriting content, so not
the default. `clean_fn`/`count_fn` are injected (test/substitution) — like `extract_fn`/`chunk_fn` in neighbors.

DI / config-agnostic: the method (the label for the param hash) is EXPLICIT; the mechanism
(clean_fn/gateway) is injected by the composition root (S13). No silent fallback chains: the param
hash and provenance are deterministic.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from functools import cached_property

from ingest.contracts import Section
from ingest.orchestrator import StepContext

CleanFn = Callable[[str], str]
CountFn = Callable[[str], int]

CLEAN_PROMPT = (
    "You are a text de-noiser for PDF-extracted prose. Fix ONLY extraction artifacts: "
    "merge words wrongly split by spaces, remove spaces before punctuation, join words "
    "hyphenated across line breaks, collapse runs of spaces. DO NOT paraphrase, summarize, "
    "translate, reorder, add, or remove any content. Return ONLY the corrected text."
)

# --- deterministic normalizer of pdfminer artifacts ------------------------------
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")            # "repli-\ncation" → "replication"
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?)\]])")  # "word ," → "word,"
_SPACE_AFTER_OPEN = re.compile(r"([(\[])[ \t]+")          # "( x" → "(x"
_MULTISPACE = re.compile(r"[ \t]{2,}")                    # collapse space runs
_TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)          # trailing spaces per line
_BLANK_RUNS = re.compile(r"\n{3,}")                       # 3+ newlines → paragraph (2)


class CleanError(RuntimeError):
    """Invalid step configuration (e.g. method='llm' without a gateway) — fail, don't go silent."""


def normalize_pdf_text(text: str) -> str:
    """Deterministic, safe cleanup of typical pdfminer artifacts. Idempotent.

    Doesn't touch single spaces between words (merging `rep lication` without a dictionary is
    unsafe) — only clearly false spaces (runs, before punctuation) and hyphen line breaks.
    """
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _MULTISPACE.sub(" ", text)
    text = _TRAILING.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


class CleanStep:
    name: str = "03_clean"
    artifacts: Sequence[str] = ("03_clean/sections.jsonl",)

    def __init__(
        self,
        *,
        method: str = "normalize",
        clean_fn: CleanFn | None = None,
        gateway=None,
        count_fn: CountFn | None = None,
        tokenizer: str = "cl100k_base",
    ) -> None:
        if method not in ("normalize", "llm"):
            raise CleanError(f"Unknown clean method {method!r}; available: normalize, llm.")
        self._method = method
        self._explicit_clean_fn = clean_fn
        self._gateway = gateway
        self._count_fn = count_fn
        self._tokenizer = tokenizer

    def params(self) -> dict:
        # Restart slice: changed clean method → hash mismatch → re-clean and everything below.
        return {"method": self._method}

    # --- orchestrator step ------------------------------------------------------
    def run(self, ctx: StepContext) -> str:
        src = ctx.rundir.path("02_structure", "sections.jsonl")
        sections = [
            Section.model_validate_json(ln)
            for ln in src.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        cleaned = self.clean_sections(sections)
        out = ctx.stage_dir / "sections.jsonl"
        out.write_text(
            "\n".join(s.model_dump_json() for s in cleaned) + "\n",
            encoding="utf-8",
        )
        return str(out)

    # --- pure cleanup (unit-testable, no disk) ---------------------------
    def clean_sections(self, sections: Sequence[Section]) -> list[Section]:
        clean_fn = self._clean_fn
        count_fn = self._count_fn or self._default_count_fn
        out: list[Section] = []
        for sec in sections:
            text = clean_fn(sec.text)
            out.append(sec.model_copy(update={"text": text, "token_count": count_fn(text)}))
        return out

    # --- resolve the cleanup mechanism -----------------------------------------------
    @property
    def _clean_fn(self) -> CleanFn:
        if self._explicit_clean_fn is not None:
            return self._explicit_clean_fn
        if self._method == "llm":
            if self._gateway is None:
                raise CleanError("method='llm' requires an injected gateway.")
            return self._llm_clean_fn(self._gateway)
        return normalize_pdf_text

    @staticmethod
    def _llm_clean_fn(gateway) -> CleanFn:
        def _clean(text: str) -> str:
            return gateway.completion(
                task="clean",
                messages=[
                    {"role": "system", "content": CLEAN_PROMPT},
                    {"role": "user", "content": text},
                ],
            )

        return _clean

    @cached_property
    def _default_count_fn(self) -> CountFn:
        import tiktoken

        enc = tiktoken.get_encoding(self._tokenizer)
        return lambda s: len(enc.encode(s))
