"""QR11 (red): composition root + query-side CLI (`rag.app` + `rag.__main__`).

Pins the contract of Rag_query_architecture.md §6/§8, Rag_implementation_steps.md (Q10):
  • `build_*` assemble `DeepSearch` FROM settings (search.* / loop.* thresholds reach the loop);
  • `model_map` routes task→model (plan/reflect/synth/embed → config values);
  • `gateway`/`connect` injection reaches the searchers/expander/loop;
  • assembly is LAZY — no network, no DB during construction (connect/gateway not called);
  • CLI parses `--question` + override flags + `--format text|json`, forwards filters.

Pure unit (CLAUDE.md §5): fake gateway/connect + build/load injection into main — no network/DB.
RED before Q10: modules `rag.app` / `rag.__main__` do not exist yet.
"""
from __future__ import annotations

import json

import pytest

from rag.contracts import DeepSearchResult, RetrievedChunk, SearchPlan


@pytest.fixture
def env(monkeypatch):
    """Minimally valid environment: secrets in env (as in QR1)."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://rag:rag@localhost:55432/rag")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    return monkeypatch


@pytest.fixture
def settings(env):
    from rag.config import load_settings

    return load_settings()


# --- fakes ----------------------------------------------------------------------------

class _Msg:
    def __init__(self, content): self.message = type("M", (), {"content": content})


class _Resp:
    def __init__(self, content): self.choices = [_Msg(content)]


class FakeConnect:
    """Stub connection factory: counts calls (to verify laziness)."""

    def __init__(self): self.calls = 0

    def __call__(self):
        self.calls += 1
        raise AssertionError("connect() must not be called during assembly")


class FakeGateway:
    def __init__(self): self.completion_calls = []

    def completion(self, *, task, messages, **kw):
        self.completion_calls.append(task)
        return "{}"


# --- build_model_map / build_gateway --------------------------------------------------

def test_build_model_map_routes_tasks(settings):
    from rag.app import build_model_map

    mm = build_model_map(settings)
    assert mm["plan"] == settings.llm.plan
    assert mm["reflect"] == settings.llm.reflect
    assert mm["synth"] == settings.llm.synth
    assert mm["embed"] == settings.embed.model   # the query embedder uses task="embed"


def test_build_gateway_resolves_task_to_model(settings):
    """Gateway from config routes task→model via model_map (One LLM gateway)."""
    seen = {}

    def fake_completion(*, model, messages, **kw):
        seen["model"] = model
        return _Resp("ok")

    from rag.app import build_gateway

    gw = build_gateway(settings, completion_fn=fake_completion)
    out = gw.completion(task="plan", messages=[{"role": "user", "content": "x"}])
    assert out == "ok"
    assert seen["model"] == settings.llm.plan      # task='plan' → llm.plan


# --- build_deep_search: settings wiring + injection + laziness ------------------------

def test_build_deep_search_wires_settings_and_injection(settings):
    from rag.app import build_deep_search

    gw = FakeGateway()
    connect = FakeConnect()
    ds = build_deep_search(settings, gateway=gw, connect=connect)

    # loop thresholds came FROM settings
    assert ds._rrf_k == int(settings.search.rrf_k)
    assert ds._mmr_lambda == float(settings.search.mmr_lambda)
    assert ds._mmr_top_n == int(settings.search.mmr_top_n)
    assert ds._relevance_threshold == float(settings.search.relevance_threshold)
    assert ds._max_chunks == int(settings.search.max_chunks)
    assert ds._max_iterations == int(settings.loop.max_iterations)
    assert ds._cost_ceiling == float(settings.loop.cost_ceiling)

    # gateway/connect injection reached the loop and all searchers/expander
    assert ds._gateway is gw
    assert ds._vector._connect is connect and ds._vector._gateway is gw
    assert ds._bm25._connect is connect
    assert ds._expander._connect is connect
    assert ds._vector._top_k == int(settings.search.top_k)
    assert ds._bm25._top_k == int(settings.search.top_k)


def test_build_deep_search_is_lazy(settings):
    """Assembly opens no connections and does not call the gateway (laziness §6)."""
    from rag.app import build_deep_search

    gw = FakeGateway()
    connect = FakeConnect()
    build_deep_search(settings, gateway=gw, connect=connect)

    assert connect.calls == 0
    assert gw.completion_calls == []


def test_build_deep_search_applies_overrides(settings):
    """CLI override beats the settings value (None in overrides is ignored)."""
    from rag.app import build_deep_search

    ds = build_deep_search(
        settings,
        gateway=FakeGateway(),
        connect=FakeConnect(),
        overrides={"top_k": 3, "max_iterations": 1, "mmr_lambda": None},
    )
    assert ds._vector._top_k == 3                      # override reached the searcher
    assert ds._max_iterations == 1                     # override reached the loop
    assert ds._mmr_lambda == float(settings.search.mmr_lambda)  # None → take from settings


def test_build_deep_search_wires_corpus_language(settings):
    """Corpus language from settings ([corpus] language) reaches the loop (cross-lingual search)."""
    from rag.app import build_deep_search

    ds = build_deep_search(settings, gateway=FakeGateway(), connect=FakeConnect())
    assert ds._corpus_language == str(settings.corpus.language)


def test_build_deep_search_corpus_language_override(settings):
    """CLI flag --corpus-language selectively overrides the corpus language."""
    from rag.app import build_deep_search

    ds = build_deep_search(
        settings, gateway=FakeGateway(), connect=FakeConnect(),
        overrides={"corpus_language": "Russian"},
    )
    assert ds._corpus_language == "Russian"


def test_build_deep_search_wires_references_penalty(settings):
    """References-list rank penalty from settings ([search] references_penalty) reaches the loop."""
    from rag.app import build_deep_search

    ds = build_deep_search(settings, gateway=FakeGateway(), connect=FakeConnect())
    assert ds._references_penalty == float(settings.search.references_penalty)


def test_build_deep_search_references_penalty_override(settings):
    """CLI flag --references-penalty selectively overrides the penalty (1.0 = disable)."""
    from rag.app import build_deep_search

    ds = build_deep_search(
        settings, gateway=FakeGateway(), connect=FakeConnect(),
        overrides={"references_penalty": 1.0},
    )
    assert ds._references_penalty == 1.0


# --- CLI ------------------------------------------------------------------------------

def _result(answer="The answer [DDIA › A]."):
    return DeepSearchResult(
        answer=answer,
        citations=["DDIA › A"],
        plan=SearchPlan(sub_questions=["s"], vector_queries=["q"], bm25_queries=[]),
        iterations=1,
    )


class FakeDeepSearch:
    def __init__(self, result): self._result = result; self.run_calls = []

    def run(self, question, *, filters=None):
        self.run_calls.append((question, filters))
        return self._result


def test_cli_parses_question_and_prints_text(capsys):
    from rag.__main__ import main

    fake = FakeDeepSearch(_result())
    rc = main(
        ["--question", "how does replication work?"],
        load=lambda: object(),
        build=lambda settings, **kw: fake,
    )
    assert rc == 0
    assert fake.run_calls == [("how does replication work?", None)]
    out = capsys.readouterr().out
    assert "The answer [DDIA › A]." in out
    assert "DDIA › A" in out                            # citations printed


def test_cli_forwards_overrides_and_filters(capsys):
    from rag.__main__ import main

    captured = {}

    def fake_build(settings, **kw):
        captured.update(kw)
        return FakeDeepSearch(_result())

    fake = None
    rc = main(
        ["-q", "x", "--top-k", "5", "--max-iterations", "2", "--document-ids", "1", "7"],
        load=lambda: object(),
        build=fake_build,
    )
    assert rc == 0
    ov = captured["overrides"]
    assert ov["top_k"] == 5 and ov["max_iterations"] == 2
    # document filter forwarded to run()
    # (taken from the last FakeDeepSearch assembled via build — there is only one)


def test_cli_forwards_document_filter_to_run():
    from rag.__main__ import main

    fake = FakeDeepSearch(_result())
    main(
        ["-q", "x", "--document-ids", "1", "7"],
        load=lambda: object(),
        build=lambda settings, **kw: fake,
    )
    assert fake.run_calls[0][1] == {"document_ids": [1, 7]}


def test_cli_json_format(capsys):
    from rag.__main__ import main

    fake = FakeDeepSearch(_result())
    rc = main(
        ["-q", "x", "--format", "json"],
        load=lambda: object(),
        build=lambda settings, **kw: fake,
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)   # valid JSON
    assert payload["answer"] == "The answer [DDIA › A]."
    assert payload["citations"] == ["DDIA › A"]


def test_cli_trace_prints_retrieved_chunks(capsys):
    """--trace prints retrieved chunks (source/score/fragment) — proof of grounding."""
    from rag.__main__ import main

    result = _result()
    result.chunks = [
        RetrievedChunk(
            chunk_id=74, content="...command query responsibility segregation (CQRS)...",
            document_id=1, section_id=9, source="DDIA › 3", score=0.0164,
        ),
        RetrievedChunk(
            chunk_id=11, content="caching frequent query results in Redis",
            document_id=2, section_id=4, source="Acing › 4", score=0.0099,
        ),
    ]
    fake = FakeDeepSearch(result)
    rc = main(
        ["-q", "x", "--trace"],
        load=lambda: object(),
        build=lambda settings, **kw: fake,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "The answer [DDIA › A]." in out                 # answer is still printed
    assert "TRACE" in out                                  # trace section present
    assert "chunk_id=74" in out and "score=0.0164" in out  # granular hit visible
    assert "command query responsibility segregation" in out  # source fragment visible
    assert "Acing › 4" in out                              # both sources


def test_cli_trace_off_by_default(capsys):
    """Without the flag the trace is not printed (output backward compatibility)."""
    from rag.__main__ import main

    result = _result()
    result.chunks = [RetrievedChunk(
        chunk_id=1, content="x", document_id=1, section_id=1, source="DDIA › 3", score=0.5,
    )]
    fake = FakeDeepSearch(result)
    main(["-q", "x"], load=lambda: object(), build=lambda settings, **kw: fake)
    assert "TRACE" not in capsys.readouterr().out
