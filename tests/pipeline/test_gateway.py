"""R3 (red): the end-to-end LLM gateway `llm_gateway.Gateway` (a LiteLLM wrapper).

Pins contract §5 Rag_pipeline_architecture.md + CLAUDE.md §2:
  • one gateway for completion + embedding; code depends on the interface, not the provider;
  • config-agnostic — values (model_map, dimension, retries) ONLY via the constructor;
  • "task → model" mapping: completion/embedding call the provider via model_map[task];
  • embedding() → vector of length `dimension`, response order = input order (sorted by index);
  • retries on transient errors; an unknown task is rejected.

Mock, NO network: the underlying litellm calls are injected into the constructor.
"""
import pytest

MODEL_MAP = {
    "default": "deepseek/deepseek-chat",
    "embed": "openai/text-embedding-3-large",
}
DIM = 1536


# --- mock responses in litellm shape ------------------------------------------------
class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _CompletionResponse:
    def __init__(self, content):
        self.choices = [_Msg(content)]
        self._hidden_params = {"response_cost": 0.0001}


def _embedding_response(vectors):
    """litellm EmbeddingResponse-like: data = list of dicts with index/embedding."""
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    return type("E", (), {"data": data})


# --- completion ----------------------------------------------------------------
def test_completion_maps_model_and_returns_text():
    from llm_gateway import Gateway

    seen = {}

    def fake_completion(*, model, messages, **kw):
        seen["model"] = model
        seen["messages"] = messages
        return _CompletionResponse("hello")

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, completion_fn=fake_completion)
    out = gw.completion(task="default", messages=[{"role": "user", "content": "hi"}])

    assert out == "hello"
    assert seen["model"] == "deepseek/deepseek-chat"  # resolved via model_map


def test_completion_logs_cost():
    from llm_gateway import Gateway

    logged = []
    gw = Gateway(
        model_map=MODEL_MAP,
        dimension=DIM,
        completion_fn=lambda **kw: _CompletionResponse("x"),
        cost_logger=lambda **kw: logged.append(kw),
    )
    gw.completion(task="default", messages=[{"role": "user", "content": "q"}])
    assert logged  # cost logged at least once


# --- embedding -----------------------------------------------------------------
def test_embedding_dimension_and_input_order():
    from llm_gateway import Gateway

    seen = {}

    def fake_embedding(*, model, input, **kw):
        seen["model"] = model
        seen["dimensions"] = kw.get("dimensions")
        # the provider returned data in SHUFFLED index order — the gateway must sort it
        vecs = [[float(i)] * DIM for i in range(len(input))]
        resp = _embedding_response(vecs)
        resp.data = list(reversed(resp.data))
        return resp

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, embedding_fn=fake_embedding)
    out = gw.embedding(texts=["a", "b", "c"])

    assert seen["model"] == "openai/text-embedding-3-large"
    assert seen["dimensions"] == DIM
    assert [len(v) for v in out] == [DIM, DIM, DIM]
    # response order == input order: vector i consists of value i
    assert out[0][0] == 0.0 and out[1][0] == 1.0 and out[2][0] == 2.0


def test_embedding_rejects_wrong_dimension():
    from llm_gateway import Gateway

    def bad_embedding(*, model, input, **kw):
        return _embedding_response([[0.0] * (DIM - 1) for _ in input])

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, embedding_fn=bad_embedding)
    with pytest.raises(ValueError):
        gw.embedding(texts=["a"])


# --- retries and mapping ----------------------------------------------------------
def test_retry_then_success():
    from llm_gateway import Gateway

    calls = {"n": 0}

    def flaky(*, model, messages, **kw):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return _CompletionResponse("ok")

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, retries=2, completion_fn=flaky)
    assert gw.completion(task="default", messages=[]) == "ok"
    assert calls["n"] == 2


def test_retry_exhausted_raises():
    from llm_gateway import Gateway

    calls = {"n": 0}

    def always_fail(*, model, messages, **kw):
        calls["n"] += 1
        raise RuntimeError("down")

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, retries=2, completion_fn=always_fail)
    with pytest.raises(RuntimeError):
        gw.completion(task="default", messages=[])
    assert calls["n"] == 3  # 1 attempt + 2 retries


def test_unknown_task_rejected():
    from llm_gateway import Gateway

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, completion_fn=lambda **kw: None)
    with pytest.raises(KeyError):
        gw.completion(task="does-not-exist", messages=[])


# --- completion_stream (red, for the SYNTH query-side SSE stream) -------------
# litellm.completion(stream=True) returns an iterator of chunks; the delta content is —
# chunk.choices[0].delta.content (may be None in service chunks).
class _Delta:
    def __init__(self, content):
        self.content = content


class _StreamChoice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content):
        self.choices = [_StreamChoice(content)]


def _stream(pieces):
    return iter([_Chunk(p) for p in pieces])


def test_completion_stream_yields_deltas():
    from llm_gateway import Gateway

    seen = {}

    def fake_stream(*, model, messages, **kw):
        seen["model"] = model
        seen["stream"] = kw.get("stream")
        return _stream(["Hel", "lo", " world"])

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, completion_fn=fake_stream)
    out = list(
        gw.completion_stream(task="default", messages=[{"role": "user", "content": "hi"}])
    )

    assert out == ["Hel", "lo", " world"]
    assert "".join(out) == "Hello world"
    assert seen["model"] == "deepseek/deepseek-chat"  # resolved via model_map
    assert seen["stream"] is True  # we ask the provider specifically for a stream


def test_completion_stream_skips_empty_deltas():
    from llm_gateway import Gateway

    def fake_stream(*, model, messages, **kw):
        return _stream(["a", None, "", "b"])  # skip service/empty deltas

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, completion_fn=fake_stream)
    assert list(gw.completion_stream(task="default", messages=[])) == ["a", "b"]


def test_completion_stream_unknown_task_rejected():
    from llm_gateway import Gateway

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, completion_fn=lambda **kw: iter(()))
    with pytest.raises(KeyError):
        list(gw.completion_stream(task="does-not-exist", messages=[]))
