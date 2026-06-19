"""ingest CLI entry (S13): `python -m ingest --source ... --external-id ... --guid ...`.

Thin wrapper over the composition root (`ingest.app`): parse run arguments, read the config
ONCE, assemble the pipe via DI and run the orchestrator by `guid`. Per-run arguments
(which document, under which external_id, into which run) live here; settings live in the config.

`source`/`external_id` describe a concrete document; `guid` keys the run (restart is keyed by it).
"""
from __future__ import annotations

import argparse
import json
import sys

from ingest.app import build_orchestrator
from ingest.config import load_settings
from ingest.structure import BOOK_PROFILES, DEFAULT_PROFILE, resolve_profile


def _force_utf8() -> None:
    """Avoid crashing on Cyrillic/arrows in the cp1252 Windows console (help/output is UTF-8)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass  # stream is no longer a TextIOWrapper (redirected) — skip


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    parser = argparse.ArgumentParser(
        prog="ingest",
        description="RAG-ingest: PDF -> Postgres/ParadeDB (Extract-Structure-Chunk-Embed-Persist).",
    )
    parser.add_argument("--source", required=True, help="path to the source document (PDF)")
    parser.add_argument(
        "--external-id", required=True, help="stable document id (e.g. 'ddia')"
    )
    parser.add_argument("--guid", required=True, help="run id (restart key)")
    parser.add_argument(
        "--profile",
        choices=sorted(BOOK_PROFILES),
        default=DEFAULT_PROFILE,
        help="structure profile by book format (§4.2.1, chapter+subsections): "
        f"{', '.join(sorted(BOOK_PROFILES))}",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="enable optional Clean (S9): cleanup of pdfminer artifacts AFTER Structure",
    )
    parser.add_argument(
        "--clean-method",
        choices=["normalize", "llm"],
        default="normalize",
        help="cleanup engine: normalize (deterministic default) / llm (via gateway)",
    )
    args = parser.parse_args(argv)

    settings = load_settings()  # the single config read (composition root)
    profile = resolve_profile(args.profile)
    orchestrator = build_orchestrator(
        settings,
        source=args.source,
        external_id=args.external_id,
        chapter_pattern=profile["chapter"],
        subsection_mode=profile["subsections"],
        opener_pattern=profile.get("opener"),  # ddia → caps opener; manning → None (fallback)
        clean=args.clean,
        clean_method=args.clean_method,
    )
    summary = orchestrator.run(args.guid)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
