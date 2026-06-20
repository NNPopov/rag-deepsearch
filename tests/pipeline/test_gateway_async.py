"""ADR-0002 L1 (red): the ASYNC surface of `llm_gateway.Gateway` — async siblings of R3.

The async siblings mirror the sync contract one-to-one (CLAUDE.md §2, §5 Rag_pipeline_architecture.md);
they exist permanently alongside the sync methods (ADR-0002: the gateway is dual-surface — sync for the
ingest CLI, async for the query+serve core):
  • acompletion()        — async sibling of completion()  (model map, retries, cost log, KeyError);
  • aembedding()         — async sibling of embedding()   (dimension, input order, ValueError);
  • acompletion_stream() — async sibling of completion_stream() (async gen of deltas, NO retries).

The underlying async litellm calls are INJECTED (`acompletion_fn` / `aembedding_fn`) — a network-free
mock here; in prod they default to `litellm.acompletion` / `litellm.aembedding`. RED before L1: the
Gateway has no a* methods / no a*_fn constructor params.

pytest-asyncio asyncio_mode=auto → `async def test_*` run without a per-test marker.
"""
import pytest

MODEL_MAP = {
    "default": "deepseek/deepseek-chat",
    "embed": "openai/text-embedding-3-large",
}
DIM = 1536


# --- mock responses in litellm shape (same shapes as the sync R3 test) --------------
class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _CompletionResponse:
    def __init__(self, content):
        self.choices = [_Msg(content)]
        self._hidden_params = {"response_cost": 0.0001}


def _embedding_response(vectors):
    data = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    return type("E", (), {"data": data})


class _AsyncStream:
    """Async iterator over pre-set chunks — what `await litellm.acompletion(stream=True)` returns."""

    def __init__(self, chunks):
        self._it = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _Delta:
    def __init__(self, content):
        self.content = content


class _StreamChoice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content):
        self.choices = [_StreamChoice(content)]


def _astream(pieces):
    return _AsyncStream([_Chunk(p) for p in pieces])


# --- acompletion ---------------------------------------------------------------------
async def test_acompletion_maps_model_and_returns_text():
    from llm_gateway import Gateway

    seen = {}

    async def fake_acompletion(*, model, messages, **kw):
        seen["model"] = model
        seen["messages"] = messages
        return _CompletionResponse("hello")

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, acompletion_fn=fake_acompletion)
    out = await gw.acompletion(task="default", messages=[{"role": "user", "content": "hi"}])

    assert out == "hello"
    assert seen["model"] == "deepseek/deepseek-chat"  # resolved via model_map


async def test_acompletion_logs_cost():
    from llm_gateway import Gateway

    logged = []

    async def fake_acompletion(**kw):
        return _CompletionResponse("x")

    gw = Gateway(
        model_map=MODEL_MAP,
        dimension=DIM,
        acompletion_fn=fake_acompletion,
        cost_logger=lambda **kw: logged.append(kw),
    )
    await gw.acompletion(task="default", messages=[{"role": "user", "content": "q"}])
    assert logged  # cost logged at least once


async def test_acompletion_unknown_task_rejected():
    from llm_gateway import Gateway

    async def fake_acompletion(**kw):
        return _CompletionResponse("x")

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, acompletion_fn=fake_acompletion)
    with pytest.raises(KeyError):
        await gw.acompletion(task="does-not-exist", messages=[])


# --- aembedding ----------------------------------------------------------------------
async def test_aembedding_dimension_and_input_order():
    from llm_gateway import Gateway

    seen = {}

    async def fake_aembedding(*, model, input, **kw):
        seen["model"] = model
        seen["dimensions"] = kw.get("dimensions")
        # the provider returned data in SHUFFLED index order — the gateway must sort it
        vecs = [[float(i)] * DIM for i in range(len(input))]
        resp = _embedding_response(vecs)
        resp.data = list(reversed(resp.data))
        return resp

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, aembedding_fn=fake_aembedding)
    out = await gw.aembedding(texts=["a", "b", "c"])

    assert seen["model"] == "openai/text-embedding-3-large"
    assert seen["dimensions"] == DIM
    assert [len(v) for v in out] == [DIM, DIM, DIM]
    # response order == input order: vector i consists of value i
    assert out[0][0] == 0.0 and out[1][0] == 1.0 and out[2][0] == 2.0


async def test_aembedding_rejects_wrong_dimension():
    from llm_gateway import Gateway

    async def bad_aembedding(*, model, input, **kw):
        return _embedding_response([[0.0] * (DIM - 1) for _ in input])

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, aembedding_fn=bad_aembedding)
    with pytest.raises(ValueError):
        await gw.aembedding(texts=["a"])


# --- retries (async) -----------------------------------------------------------------
async def test_acompletion_retry_then_success():
    from llm_gateway import Gateway

    calls = {"n": 0}

    async def flaky(*, model, messages, **kw):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return _CompletionResponse("ok")

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, retries=2, acompletion_fn=flaky)
    assert await gw.acompletion(task="default", messages=[]) == "ok"
    assert calls["n"] == 2


async def test_acompletion_retry_exhausted_raises():
    from llm_gateway import Gateway

    calls = {"n": 0}

    async def always_fail(*, model, messages, **kw):
        calls["n"] += 1
        raise RuntimeError("down")

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, retries=2, acompletion_fn=always_fail)
    with pytest.raises(RuntimeError):
        await gw.acompletion(task="default", messages=[])
    assert calls["n"] == 3  # 1 attempt + 2 retries


# --- acompletion_stream (async generator, for the SYNTH SSE stream) ------------------
async def test_acompletion_stream_yields_deltas():
    from llm_gateway import Gateway

    seen = {}

    async def fake_astream(*, model, messages, **kw):
        seen["model"] = model
        seen["stream"] = kw.get("stream")
        return _astream(["Hel", "lo", " world"])

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, acompletion_fn=fake_astream)
    out = [
        d
        async for d in gw.acompletion_stream(
            task="default", messages=[{"role": "user", "content": "hi"}]
        )
    ]

    assert out == ["Hel", "lo", " world"]
    assert "".join(out) == "Hello world"
    assert seen["model"] == "deepseek/deepseek-chat"  # resolved via model_map
    assert seen["stream"] is True  # we ask the provider specifically for a stream


async def test_acompletion_stream_skips_empty_deltas():
    from llm_gateway import Gateway

    async def fake_astream(*, model, messages, **kw):
        return _astream(["a", None, "", "b"])  # skip service/empty deltas

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, acompletion_fn=fake_astream)
    out = [d async for d in gw.acompletion_stream(task="default", messages=[])]
    assert out == ["a", "b"]


async def test_acompletion_stream_unknown_task_rejected():
    from llm_gateway import Gateway

    async def fake_astream(**kw):
        return _astream(())

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, acompletion_fn=fake_astream)
    with pytest.raises(KeyError):
        # lazy: KeyError surfaces at the first iteration (mirrors the sync stream contract)
        [d async for d in gw.acompletion_stream(task="does-not-exist", messages=[])]


async def test_acompletion_stream_no_retries():
    """A stream can't be replayed from the middle → the streaming path does NOT retry (mirrors sync)."""
    from llm_gateway import Gateway

    calls = {"n": 0}

    async def failing_open(*, model, messages, **kw):
        calls["n"] += 1
        raise RuntimeError("open failed")

    gw = Gateway(model_map=MODEL_MAP, dimension=DIM, retries=2, acompletion_fn=failing_open)
    with pytest.raises(RuntimeError):
        [d async for d in gw.acompletion_stream(task="default", messages=[])]
    assert calls["n"] == 1  # opened once, no retry
