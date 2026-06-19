"""Extract filter (S7) — the first pipeline stage.

Single responsibility (§1, §4.1 Rag_pipeline_architecture.md): get flat text from the
source PDF and write it to `01_extract/full_text.txt`. Markers are NOT cleaned out (that's
Structure). Implements the orchestrator step contract (S6): `run(ctx) -> artifact`.

DI and config-agnostic: the source and method come via the constructor (composition root, S13).
The backend function can be injected (test/substitution), otherwise resolved from `BACKENDS[method]`.
The method is an EXPLICIT parameter, with no silent fallback chain: param hash and provenance are deterministic.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence

from ingest.extract.backends import BACKENDS
from ingest.orchestrator import StepContext


class ExtractError(RuntimeError):
    """The backend gave no usable text (empty/None) — the step must fail, not write a stub."""


class ExtractStep:
    name: str = "01_extract"
    artifacts: Sequence[str] = ("01_extract/full_text.txt",)

    def __init__(
        self,
        *,
        source: str | os.PathLike,
        method: str = "pdfminer",
        extract_fn: Callable[[str], str | None] | None = None,
    ) -> None:
        self._source = str(source)
        self._method = method
        self._extract_fn = extract_fn

    def params(self) -> dict:
        # Restart slice: changed source or method → hash mismatch → re-extract.
        return {"source": self._source, "method": self._method}

    def run(self, ctx: StepContext) -> str:
        text = self._extract()
        if not text or not text.strip():
            raise ExtractError(
                f"Extract produced empty text (source={self._source!r}, method={self._method!r})."
            )
        out = ctx.stage_dir / "full_text.txt"
        out.write_text(text, encoding="utf-8")
        return str(out)

    def _extract(self) -> str | None:
        fn = self._extract_fn
        if fn is None:
            try:
                fn = BACKENDS[self._method]
            except KeyError:
                raise ExtractError(
                    f"Unknown extraction method {self._method!r}; "
                    f"available: {sorted(BACKENDS)}."
                ) from None
        return fn(self._source)
