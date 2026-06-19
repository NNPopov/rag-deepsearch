"""QR1 (red): query-side config `rag` (Dynaconf) — composition root.

Pins the contract of Rag_query_architecture.md §6 + §8 (Q0):
  • config is read by ONE factory `rag.config.load_settings()` (single read point);
  • secrets bound from env (DATABASE_URL / OPENAI_API_KEY / DEEPSEEK_API_KEY);
  • envvar_prefix="RAG" — a RAG_* variable overrides a setting;
  • threshold defaults in place (search.top_k/rrf_k/mmr_*/relevance_threshold/max_chunks,
    loop.max_iterations); model_map sources (llm.plan/reflect/synth, embed.model);
  • embed.dimensions == db.DIMENSION — the contract dimension, matches the halfvec(N) schema.

Pure unit, no network/DB. RED before Q0: package `rag.config` does not exist yet.
"""
import pytest


@pytest.fixture
def env(monkeypatch):
    """Minimally valid environment: secrets set in env."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:55432/rag")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    return monkeypatch


def test_secrets_bound_from_env(env):
    from rag.config import load_settings

    s = load_settings()
    assert s.database_url == "postgresql://rag:rag@localhost:55432/rag"
    assert s.openai_api_key == "sk-openai-test"
    assert s.deepseek_api_key == "sk-deepseek-test"


def test_dimensions_matches_db_constant(env):
    import db
    from rag.config import load_settings

    s = load_settings()
    assert int(s.embed.dimensions) == db.DIMENSION == 1536


def test_model_map_sources_present(env):
    """model_map sources: tasks plan/reflect/synth + the embedder model."""
    from rag.config import load_settings

    s = load_settings()
    assert s.llm.plan and s.llm.reflect and s.llm.synth
    assert s.embed.model


def test_search_defaults_present(env):
    from rag.config import load_settings

    s = load_settings()
    assert int(s.search.top_k) > 0
    assert int(s.search.rrf_k) == 60
    assert 0.0 <= float(s.search.mmr_lambda) <= 1.0
    assert int(s.search.mmr_top_n) > 0
    assert float(s.search.relevance_threshold) >= 0.0
    assert int(s.search.max_chunks) > 0


def test_loop_defaults_present(env):
    from rag.config import load_settings

    s = load_settings()
    assert int(s.loop.max_iterations) > 0


def test_envvar_prefix_override(env):
    from rag.config import load_settings

    env.setenv("RAG_LOOP__MAX_ITERATIONS", "7")
    s = load_settings()
    assert int(s.loop.max_iterations) == 7


def test_missing_secret_is_detectable(monkeypatch):
    """Without DATABASE_URL in env, accessing the secret must not "silently" return a valid DSN."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    from rag.config import load_settings

    s = load_settings()
    with pytest.raises(Exception):
        _ = s.database_url
