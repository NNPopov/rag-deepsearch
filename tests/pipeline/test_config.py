"""R9 (red): ingest config (Dynaconf) — composition root.

Pins contract §7 Rag_pipeline_architecture.md:
  • config is read by ONE factory `load_settings()` (single point of reading);
  • secrets are bound from env (DATABASE_URL / OPENAI_API_KEY / DEEPSEEK_API_KEY);
  • envvar_prefix="INGEST" — an INGEST_* variable overrides a setting;
  • non-secret defaults are in place (embed.dimensions, chunk.chunk_size/overlap, run_dir);
  • embed.dimensions == db.DIMENSION — the contract dimension, matches the schema.

Pure unit, no network/DB. RED until S3: the ingest.config package does not exist yet.
"""
import pytest


@pytest.fixture
def env(monkeypatch):
    """Minimally valid environment: secrets are set in env."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:55432/rag")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    return monkeypatch


def test_secrets_bound_from_env(env):
    from ingest.config import load_settings

    s = load_settings()
    assert s.database_url == "postgresql://rag:rag@localhost:55432/rag"
    assert s.openai_api_key == "sk-openai-test"
    assert s.deepseek_api_key == "sk-deepseek-test"


def test_dimensions_matches_db_constant(env):
    import db
    from ingest.config import load_settings

    s = load_settings()
    assert int(s.embed.dimensions) == db.DIMENSION == 1536


def test_chunk_defaults_present(env):
    from ingest.config import load_settings

    s = load_settings()
    assert int(s.chunk.chunk_size) > 0
    assert int(s.chunk.overlap) >= 0
    assert int(s.chunk.overlap) < int(s.chunk.chunk_size)


def test_run_dir_default_present(env):
    from ingest.config import load_settings

    s = load_settings()
    assert s.run_dir  # non-empty default path


def test_envvar_prefix_override(env):
    from ingest.config import load_settings

    env.setenv("INGEST_RUN_DIR", "custom_runs_dir")
    s = load_settings()
    assert s.run_dir == "custom_runs_dir"


def test_missing_secret_is_detectable(monkeypatch):
    """Without DATABASE_URL in env, accessing the secret must not "silently" return a valid DSN."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    from ingest.config import load_settings

    s = load_settings()
    with pytest.raises(Exception):
        _ = s.database_url
