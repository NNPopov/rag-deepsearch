"""R11 (red): Embed filter `ingest.embed.EmbedStep` (S11).

Contract §4.4 Rag_pipeline_architecture.md: read `03_chunk/chunks.jsonl` (`Chunk`),
vectorize the text THROUGH `llm_gateway` (LiteLLM `embedding`) and write
`04_embed/chunks_embedded.jsonl` — one `EmbeddedChunk` per line (Embed→Persist pipe).
We check: each chunk gets a vector of length `db.DIMENSION`, response order = input order,
batches by token budget (`token_count` already computed at S10) and by batch size limit,
`embedding_model` (provenance) and all `Chunk` contract fields carried through.

The gateway is INJECTED (config-agnostic) → mock with no network, a pure deterministic unit.
RED before S11: the ingest.embed module does not exist yet.
"""
from __future__ import annotations

import pytest

import db
from ingest.contracts import Chunk, EmbeddedChunk
from ingest.orchestrator import Orchestrator
from ingest.rundir import RunDir


class FakeGateway:
    """Stand-in for llm_gateway.Gateway: records batch order, returns a vector of length DIMENSION.

    The vector's first element = len(text) → the test uses it to check that the right vector
    is bound to the right chunk (mapping + order). No network, deterministic.
    """

    def __init__(self, dim: int = db.DIMENSION) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []
        self.tasks: list[str] = []

    def embedding(self, *, texts, task: str = "embed") -> list[list[float]]:
        self.calls.append(list(texts))
        self.tasks.append(task)
        return [[float(len(t))] + [0.0] * (self.dim - 1) for t in texts]


def _chunk(section_path: str, seq: int, text: str, *, token_count: int | None = None) -> Chunk:
    return Chunk(
        external_id="ddia",
        section_path=section_path,
        chapter=int(section_path.split(".")[0]),
        heading="Some Heading",
        seq=seq,
        text=text,
        token_count=token_count if token_count is not None else len(text.split()),
        page_range=(149, 149),
        metadata={"k": "v"},
    )


@pytest.fixture()
def step():
    from ingest.embed import EmbedStep

    return EmbedStep(
        gateway=FakeGateway(),
        embedding_model="text-embedding-3-large",
        max_batch_tokens=10_000,
        max_batch_size=2048,
    )


# --- Step contract (S6) --------------------------------------------------------------------
def test_conforms_to_step_protocol(step):
    from ingest.orchestrator import Step

    assert isinstance(step, Step)
    assert step.name == "04_embed"
    assert "04_embed/chunks_embedded.jsonl" in step.artifacts


def test_params_include_model_and_batch_knobs(step):
    p = step.params()
    assert p["embedding_model"] == "text-embedding-3-large"
    assert p["max_batch_tokens"] == 10_000
    assert p["max_batch_size"] == 2048
    assert "task" in p  # changing any → invalidation (S6 cascade)


# --- each chunk → EmbeddedChunk with a vector of length DIMENSION, contract carried through ---------------
def test_embeds_each_chunk_and_preserves_contract():
    from ingest.embed import EmbedStep

    gw = FakeGateway()
    st = EmbedStep(gateway=gw, embedding_model="text-embedding-3-large")
    chunks = [_chunk("4.2", 0, "alpha beta"), _chunk("4.2", 1, "gamma")]
    out = st.embed_chunks(chunks)

    assert len(out) == 2
    for c_in, ec in zip(chunks, out):
        assert isinstance(ec, EmbeddedChunk)
        assert len(ec.vector) == db.DIMENSION
        assert ec.embedding_model == "text-embedding-3-large"
        # Chunk contract fields carried through unaltered
        assert ec.section_path == c_in.section_path
        assert ec.chapter == c_in.chapter
        assert ec.seq == c_in.seq
        assert ec.text == c_in.text
        assert ec.page_range == c_in.page_range
        assert ec.metadata == c_in.metadata
        assert ec.id == c_in.id  # computed id survives vectorization


# --- response order = input order --------------------------------------------------------
def test_order_preserved_across_batches():
    from ingest.embed import EmbedStep

    gw = FakeGateway()
    # small budget → several batches; vector[0]=len(text) must stay with its chunk
    st = EmbedStep(gateway=gw, embedding_model="m", max_batch_tokens=120)
    chunks = [_chunk("1.1", i, "x" * (i + 1), token_count=50) for i in range(5)]
    out = st.embed_chunks(chunks)

    assert [ec.seq for ec in out] == [0, 1, 2, 3, 4]
    assert [int(ec.vector[0]) for ec in out] == [len(c.text) for c in chunks]


# --- batches by token budget (token_count from S10) -----------------------------------------
def test_batches_by_token_budget():
    from ingest.embed import EmbedStep

    gw = FakeGateway()
    st = EmbedStep(gateway=gw, embedding_model="m", max_batch_tokens=250, max_batch_size=2048)
    chunks = [_chunk("1.1", i, f"chunk {i}", token_count=100) for i in range(4)]
    st.embed_chunks(chunks)

    # 100+100=200 ≤ 250, +100=300 > 250 → cut; result is 2 batches of 2
    assert [len(b) for b in gw.calls] == [2, 2]
    for batch in gw.calls:
        assert len(batch) <= 250 // 100  # batch token_count sum within budget


def test_oversized_single_chunk_goes_alone():
    from ingest.embed import EmbedStep

    gw = FakeGateway()
    st = EmbedStep(gateway=gw, embedding_model="m", max_batch_tokens=100)
    chunks = [_chunk("1.1", 0, "big", token_count=500), _chunk("1.1", 1, "small", token_count=20)]
    out = st.embed_chunks(chunks)

    # a chunk larger than the budget is not lost — it goes as its own batch
    assert [len(b) for b in gw.calls] == [1, 1]
    assert len(out) == 2


# --- batches by size limit ---------------------------------------------------------------
def test_batches_by_max_size():
    from ingest.embed import EmbedStep

    gw = FakeGateway()
    st = EmbedStep(gateway=gw, embedding_model="m", max_batch_tokens=10_000, max_batch_size=2)
    chunks = [_chunk("1.1", i, f"c{i}", token_count=1) for i in range(5)]
    st.embed_chunks(chunks)

    assert [len(b) for b in gw.calls] == [2, 2, 1]


# --- works as an orchestrator step: reads chunks.jsonl, writes chunks_embedded.jsonl --------
def test_runs_as_orchestrator_step(tmp_path):
    from ingest.embed import EmbedStep

    base = tmp_path / "runs"
    guid = "g-embed"
    rd = RunDir(base, guid)
    chunk_dir = rd.stage_dir("03_chunk")
    chunks = [_chunk("1.1", 0, "alpha beta gamma"), _chunk("1.1", 1, "delta")]
    (chunk_dir / "chunks.jsonl").write_text(
        "\n".join(c.model_dump_json() for c in chunks) + "\n", encoding="utf-8"
    )

    step = EmbedStep(gateway=FakeGateway(), embedding_model="text-embedding-3-large")
    Orchestrator([step], base=base).run(guid)

    out = rd.path("04_embed", "chunks_embedded.jsonl")
    assert out.exists()
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    parsed = [EmbeddedChunk.model_validate_json(ln) for ln in lines]
    assert len(parsed) == 2
    assert all(len(ec.vector) == db.DIMENSION for ec in parsed)
    assert all(ec.embedding_model == "text-embedding-3-large" for ec in parsed)
    assert [ec.id for ec in parsed] == [c.id for c in chunks]


def test_empty_input_yields_no_chunks():
    from ingest.embed import EmbedStep

    gw = FakeGateway()
    st = EmbedStep(gateway=gw, embedding_model="m")
    assert st.embed_chunks([]) == []
    assert gw.calls == []  # empty input → gateway is not called
