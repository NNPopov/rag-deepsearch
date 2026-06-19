"""ingest config (Dynaconf) — composition root, the single read point (§7 of the architecture).

Contract:
  • `load_settings()` — the ONLY config-reading factory; everything else gets its
    values via constructors (DI). No reads of env/Dynaconf scattered around the code.
  • Secrets (DATABASE_URL / OPENAI_API_KEY / DEEPSEEK_API_KEY) bind from env by name;
    the values never land in git.
  • `envvar_prefix="INGEST"` — INGEST_* overrides any setting (e.g. INGEST_RUN_DIR,
    INGEST_CHUNK__CHUNK_SIZE).
  • `embed.dimensions` must match the contract `db.DIMENSION` (halfvec(N)) — otherwise an error.
"""
from pathlib import Path

from dynaconf import Dynaconf

import db

_HERE = Path(__file__).parent
_SETTINGS_FILES = [str(_HERE / "settings.toml"), str(_HERE / ".secrets.toml")]


def load_settings() -> Dynaconf:
    """Assembles and validates the ingest config. Call ONCE in the composition root."""
    settings = Dynaconf(
        envvar_prefix="INGEST",
        settings_files=_SETTINGS_FILES,
        load_dotenv=True,
    )
    _validate(settings)
    return settings


def _validate(settings: Dynaconf) -> None:
    """Config contract invariants (don't touch secrets — those resolve lazily)."""
    dim = int(settings.embed.dimensions)
    if dim != db.DIMENSION:
        raise ValueError(
            f"embed.dimensions ({dim}) != db.DIMENSION ({db.DIMENSION}); "
            "the config dimension must match the halfvec(N) schema."
        )
