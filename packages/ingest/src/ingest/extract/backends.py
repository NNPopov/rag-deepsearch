"""Backends for extracting flat text from a PDF (S7).

Absorbed from the standalone extractor (`scripts/extractor/parsers/pdf.py`, Variant 2 §6.265).
Each function: `(pdf_path) -> str | None` — the text or None if the backend is unavailable / gave no
result. Markers (CHAPTER N, footers, \f page breaks) are NOT cleaned out — that's the job of
Structure (S8). We do NOT take docling here (see §7 CLAUDE.md: bad_alloc on Windows).

The method is chosen by an explicit filter parameter, not a "silent" fallback chain: the step's
provenance and param hash must be deterministic. Default is pdfminer (it produced a working full_text.txt).
"""
from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable


def extract_with_pdfminer(pdf_path: str) -> str | None:
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        return None
    return extract_text(pdf_path)


def extract_with_pypdf2(pdf_path: str) -> str | None:
    try:
        import PyPDF2
    except ImportError:
        return None
    parts: list[str] = []
    with open(pdf_path, "rb") as f:
        for page in PyPDF2.PdfReader(f).pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
    return "\n".join(parts)


def extract_with_pdftotext(pdf_path: str) -> str | None:
    if not shutil.which("pdftotext"):  # poppler-utils; usually absent on Windows
        return None
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout
    return None


# Registry of available methods; key = the `method` value in the filter config.
BACKENDS: dict[str, Callable[[str], str | None]] = {
    "pdfminer": extract_with_pdfminer,
    "pypdf2": extract_with_pypdf2,
    "pdftotext": extract_with_pdftotext,
}
