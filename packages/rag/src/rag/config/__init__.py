"""rag config (Dynaconf) — composition root, the single read point (Rag_query_architecture.md §6).

Contract:
  • `load_settings()` — the ONLY config-reading factory; everything else receives
    values through constructors (DI). No scattered env/Dynaconf reads across the code.
  • Secrets (DATABASE_URL / OPENAI_API_KEY / DEEPSEEK_API_KEY) are bound from env by name;
    the values never land in git.
  • `envvar_prefix="RAG"` — RAG_* overrides any setting (e.g. RAG_LOOP__MAX_ITERATIONS,
    RAG_SEARCH__TOP_K).
  • `embed.dimensions` must match the contract `db.DIMENSION` (halfvec(N)) — otherwise an error.
"""
from pathlib import Path

from dynaconf import Dynaconf

import db

_HERE = Path(__file__).parent
_SETTINGS_FILES = [str(_HERE / "settings.toml"), str(_HERE / ".secrets.toml")]


def load_settings() -> Dynaconf:
    """Assembles and validates the rag config. Call ONCE in the composition root."""
    settings = Dynaconf(
        envvar_prefix="RAG",
        settings_files=_SETTINGS_FILES,
        load_dotenv=True,
    )
    _validate(settings)
    return settings


def _validate(settings: Dynaconf) -> None:
    """Config contract invariants (do not touch secrets — those resolve lazily)."""
    dim = int(settings.embed.dimensions)
    if dim != db.DIMENSION:
        raise ValueError(
            f"embed.dimensions ({dim}) != db.DIMENSION ({db.DIMENSION}); "
            "the config dimension must match the schema's halfvec(N)."
        )
