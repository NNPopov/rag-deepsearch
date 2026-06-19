"""R12 (red): composition root `ingest.app` (S13).

Contract §7 Rag_pipeline_architecture.md: config is read in ONE place and passed down
ONLY through constructors (DI). `build_orchestrator(settings, source, external_id)` assembles
the whole Extract→Structure→Chunk→Embed→Persist pipeline, threading values from `settings` into the steps;
shared `gateway`/`connect` are built here and injected (config-agnostic — they don't read config themselves).

Assembly is LAZY: no network, no DB at build time (litellm/psycopg are touched only at run). Hence
the test is a pure unit — `load_settings()` reads non-secret defaults from settings.toml, secrets
(`@format {env[...]}`) are left untouched. RED until S13: the ingest.app module does not exist yet.
"""
from __future__ import annotations

import db
from ingest.config import load_settings


def _settings():
    return load_settings()


# --- pipeline assembled in the correct order -----------------------------------------------
def test_pipeline_steps_in_order():
    from ingest.app import build_orchestrator

    orch = build_orchestrator(_settings(), source="book.pdf", external_id="ddia")
    assert [s.name for s in orch.steps] == [
        "01_extract", "02_structure", "03_chunk", "04_embed", "05_persist",
    ]


def test_orchestrator_base_is_run_dir():
    from ingest.app import build_orchestrator

    settings = _settings()
    orch = build_orchestrator(settings, source="book.pdf", external_id="ddia")
    assert str(orch.base) == str(settings.run_dir)


# --- settings values reach the steps (DI, single point of reading) -------------------------
def test_chunk_step_uses_settings():
    from ingest.app import build_orchestrator

    settings = _settings()
    orch = build_orchestrator(settings, source="book.pdf", external_id="ddia")
    chunk = next(s for s in orch.steps if s.name == "03_chunk")
    assert chunk.params()["chunk_size"] == settings.chunk.chunk_size
    assert chunk.params()["overlap"] == settings.chunk.overlap


def test_embed_step_uses_settings():
    from ingest.app import build_orchestrator

    settings = _settings()
    orch = build_orchestrator(settings, source="book.pdf", external_id="ddia")
    embed = next(s for s in orch.steps if s.name == "04_embed")
    p = embed.params()
    assert p["embedding_model"] == settings.embed.model
    assert p["max_batch_tokens"] == settings.embed.max_batch_tokens
    assert p["max_batch_size"] == settings.embed.max_batch_size


def test_extract_and_structure_wired():
    from ingest.app import build_orchestrator

    orch = build_orchestrator(_settings(), source="book.pdf", external_id="ddia")
    extract = next(s for s in orch.steps if s.name == "01_extract")
    structure = next(s for s in orch.steps if s.name == "02_structure")
    assert extract.params()["source"].endswith("book.pdf")
    assert structure.params()["external_id"] == "ddia"


def test_structure_uses_default_profile_when_unset():
    """Without an override, Structure takes the default (ddia/footers) profile — backward compatibility."""
    from ingest.app import build_orchestrator
    from ingest.structure import BOOK_PROFILES

    orch = build_orchestrator(_settings(), source="book.pdf", external_id="ddia")
    structure = next(s for s in orch.steps if s.name == "02_structure")
    assert structure.params()["chapter_pattern"] == BOOK_PROFILES["ddia"]["chapter"]
    assert structure.params()["subsection_mode"] == "footers"


def test_profile_override_reaches_structure_step():
    """Per-book profile (manning: running header + numbered) reaches StructureStep (DI)."""
    from ingest.app import build_orchestrator
    from ingest.structure import BOOK_PROFILES

    manning = BOOK_PROFILES["manning"]
    orch = build_orchestrator(
        _settings(), source="acing.pdf", external_id="acing",
        chapter_pattern=manning["chapter"], subsection_mode=manning["subsections"],
    )
    structure = next(s for s in orch.steps if s.name == "02_structure")
    assert structure.params()["chapter_pattern"] == manning["chapter"]
    assert structure.params()["subsection_mode"] == "numbered"


# --- optional Clean (S9): off by default; on inserts the step and switches the input -------
def test_clean_disabled_by_default_chunk_persist_read_structure():
    from ingest.app import build_orchestrator

    orch = build_orchestrator(_settings(), source="book.pdf", external_id="ddia")
    names = [s.name for s in orch.steps]
    assert "03_clean" not in names
    chunk = next(s for s in orch.steps if s.name == "03_chunk")
    persist = next(s for s in orch.steps if s.name == "05_persist")
    assert chunk.params()["input_stage"] == "02_structure"
    assert persist.params()["sections_stage"] == "02_structure"


def test_clean_enabled_inserts_step_after_structure_and_rewires_input():
    from ingest.app import build_orchestrator

    orch = build_orchestrator(
        _settings(), source="book.pdf", external_id="ddia", clean=True, clean_method="normalize"
    )
    names = [s.name for s in orch.steps]
    # Clean sits strictly between Structure and Chunk
    assert names == ["01_extract", "02_structure", "03_clean", "03_chunk", "04_embed", "05_persist"]
    clean = next(s for s in orch.steps if s.name == "03_clean")
    chunk = next(s for s in orch.steps if s.name == "03_chunk")
    persist = next(s for s in orch.steps if s.name == "05_persist")
    assert clean.params()["method"] == "normalize"
    # Chunk and Persist now read the cleaned sections
    assert chunk.params()["input_stage"] == "03_clean"
    assert persist.params()["sections_stage"] == "03_clean"


# --- gateway: task→model mapping from config ------------------------------------------------
def test_model_map_routes_embed_task():
    from ingest.app import build_model_map

    settings = _settings()
    model_map = build_model_map(settings)
    assert model_map["embed"] == settings.embed.model  # EmbedStep calls task="embed"


def test_build_gateway_carries_contract_dimension():
    from llm_gateway import Gateway

    from ingest.app import build_gateway

    settings = _settings()
    gw = build_gateway(settings)
    assert isinstance(gw, Gateway)
    # the embedding is of length db.DIMENSION: stub the lower litellm call, no network
    captured = {}

    def fake_embedding(*, model, input, dimensions):
        captured["model"] = model
        captured["dimensions"] = dimensions
        return type("R", (), {"data": [{"index": i, "embedding": [0.0] * dimensions}
                                       for i, _ in enumerate(input)]})()

    gw._embedding_fn = fake_embedding  # injection into the built gateway (as in R3)
    vecs = gw.embedding(texts=["hello"], task="embed")
    assert captured["dimensions"] == db.DIMENSION
    assert captured["model"] == settings.embed.model
    assert len(vecs[0]) == db.DIMENSION


# --- injection of shared dependencies reaches the steps (DI all the way) -------------------
def test_injected_gateway_reaches_embed_step():
    from ingest.app import build_orchestrator
    from ingest.contracts import Chunk

    class FakeGateway:
        def __init__(self):
            self.called = False

        def embedding(self, *, texts, task="embed"):
            self.called = True
            return [[0.0] * db.DIMENSION for _ in texts]

    fake = FakeGateway()
    orch = build_orchestrator(
        _settings(), source="book.pdf", external_id="ddia", gateway=fake
    )
    embed = next(s for s in orch.steps if s.name == "04_embed")
    embed.embed_chunks(
        [Chunk(external_id="ddia", section_path="1.1", chapter=1, heading="H",
               seq=0, text="x", token_count=1)]
    )
    assert fake.called  # the built EmbedStep really uses the injected gateway


def test_injected_connect_reaches_persist_step():
    from ingest.app import build_orchestrator

    sentinel = object()
    orch = build_orchestrator(
        _settings(), source="book.pdf", external_id="ddia", connect=lambda: sentinel
    )
    persist = next(s for s in orch.steps if s.name == "05_persist")
    assert persist._connect() is sentinel  # the connection factory reached the step
