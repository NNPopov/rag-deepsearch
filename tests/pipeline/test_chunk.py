"""R7 (red): Chunk filter `ingest.chunk.ChunkStep` (S10).

Contract §4.3 Rag_pipeline_architecture.md: we slice LEAF sections (`is_leaf`) from
`02_structure/sections.jsonl` into chunks (Chonkie `RecursiveChunker` + tiktoken cl100k) and
write `03_chunk/chunks.jsonl` — one valid `Chunk` per line (the Chunk→Embed pipe).
We check: `token_count ≤ chunk_size (+overlap)`, overlap really overlaps neighbors,
section metadata is carried through, `seq` numbers within a section, non-leaf chapters are NOT chunked.

The cl100k_base tokenizer matches the text-embedding-3-large embedder → we compute `token_count`
ourselves via tiktoken (the chonkie count differs). Pure unit: chonkie/tiktoken are local,
deterministic, no network. RED until S10: the ingest.chunk module does not exist yet.
"""
from __future__ import annotations

import pytest

from ingest.contracts import Chunk, Section
from ingest.orchestrator import Orchestrator
from ingest.rundir import RunDir


def _leaf(path: str, heading: str, text: str, *, page_range=None) -> Section:
    return Section(
        external_id="ddia",
        path=path,
        depth=2 if "." in path else 1,
        parent_path=path.split(".")[0] if "." in path else None,
        ordinal=int(path.split(".")[-1]) if "." in path else int(path),
        heading=heading,
        is_leaf=True,
        text=text,
        token_count=len(text.split()),
        page_range=page_range,
    )


def _chapter_nonleaf(path: str, heading: str) -> Section:
    return Section(
        external_id="ddia",
        path=path,
        depth=1,
        parent_path=None,
        ordinal=int(path),
        heading=heading,
        is_leaf=False,
        text="big block of the whole chapter " * 5,
        token_count=30,
    )


LONG = "Sentence number {i} talks about distributed systems and data. "
LONG_TEXT = "".join(LONG.format(i=i) for i in range(60)).strip()


@pytest.fixture()
def step():
    from ingest.chunk import ChunkStep

    return ChunkStep(chunk_size=64, overlap=0)


# --- Step contract (S6) --------------------------------------------------------------------
def test_conforms_to_step_protocol(step):
    from ingest.orchestrator import Step

    assert isinstance(step, Step)
    assert step.name == "03_chunk"
    assert "03_chunk/chunks.jsonl" in step.artifacts


def test_params_include_chunk_size_overlap_tokenizer(step):
    p = step.params()
    assert p["chunk_size"] == 64
    assert p["overlap"] == 0
    assert "tokenizer" in p  # changing any → invalidation (S6 cascade)


# --- only leaves are chunked ---------------------------------------------------------------
def test_only_leaf_sections_are_chunked():
    from ingest.chunk import ChunkStep

    # fake chunker: one section → one piece (we test exactly the leaf selection/wiring)
    st = ChunkStep(chunk_size=64, overlap=0, chunk_fn=lambda t: [t])
    sections = [
        _chapter_nonleaf("1", "Replication"),  # NOT a leaf → skip
        _leaf("1.1", "Single-Leader", "alpha beta gamma"),
        _leaf("1.2", "Multi-Leader", "delta epsilon"),
    ]
    chunks = st.chunk_sections(sections)
    paths = {c.section_path for c in chunks}
    assert paths == {"1.1", "1.2"}  # chapter "1" not chunked


# --- metadata wiring + seq + contract ------------------------------------------------------
def test_chunk_metadata_seq_and_contract():
    from ingest.chunk import ChunkStep

    st = ChunkStep(chunk_size=64, overlap=0, chunk_fn=lambda t: t.split("||"))
    sec = _leaf("4.2", "Multidimensional Indexes", "p0||p1||p2", page_range=(149, 149))
    chunks = st.chunk_sections([sec])
    assert [c.seq for c in chunks] == [0, 1, 2]
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.external_id == "ddia"
        assert c.section_path == "4.2"
        assert c.chapter == 4  # derived from the top of path
        assert c.heading == "Multidimensional Indexes"
        assert c.page_range == (149, 149)  # carried through from the section
        assert c.token_count > 0
        assert c.id.startswith("ddia:4.2:")  # computed id from contracts


# --- token_count ≤ chunk_size (real chonkie+tiktoken) --------------------------------------
def test_token_count_within_chunk_size(step):
    chunks = step.chunk_sections([_leaf("1.1", "Big Section", LONG_TEXT)])
    assert len(chunks) >= 2  # the long text is really sliced
    assert all(c.token_count <= 64 for c in chunks)  # overlap=0 → strictly ≤ chunk_size


def test_overlap_makes_adjacent_chunks_share_tokens():
    from ingest.chunk import ChunkStep

    words = " ".join(f"word{i}" for i in range(220))
    no_ov = ChunkStep(chunk_size=48, overlap=0).chunk_sections([_leaf("1.1", "S", words)])
    with_ov = ChunkStep(chunk_size=48, overlap=12).chunk_sections([_leaf("1.1", "S", words)])

    def shared(a: Chunk, b: Chunk) -> set[str]:
        return set(a.text.split()) & set(b.text.split())

    assert shared(no_ov[0], no_ov[1]) == set()  # without overlap neighbors don't intersect
    assert shared(with_ov[0], with_ov[1])  # with overlap — they intersect
    assert all(c.token_count <= 48 + 12 for c in with_ov)  # ceiling = chunk_size+overlap


# --- works as an orchestrator step: reads sections.jsonl, writes chunks.jsonl -------------
def test_runs_as_orchestrator_step(tmp_path):
    from ingest.chunk import ChunkStep

    base = tmp_path / "runs"
    guid = "g-chunk"
    rd = RunDir(base, guid)
    struct_dir = rd.stage_dir("02_structure")
    sections = [
        _chapter_nonleaf("1", "Replication"),
        _leaf("1.1", "Single-Leader", LONG_TEXT, page_range=(227, 227)),
    ]
    (struct_dir / "sections.jsonl").write_text(
        "\n".join(s.model_dump_json() for s in sections) + "\n", encoding="utf-8"
    )

    Orchestrator([ChunkStep(chunk_size=64, overlap=8)], base=base).run(guid)

    out = rd.path("03_chunk", "chunks.jsonl")
    assert out.exists()
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    parsed = [Chunk.model_validate_json(ln) for ln in lines]
    assert parsed and all(c.section_path == "1.1" for c in parsed)  # only the leaf got in
    assert all(c.token_count <= 64 + 8 for c in parsed)


def test_empty_leaf_yields_no_chunks(step):
    chunks = step.chunk_sections([_leaf("9.0", "(preamble)", "   ")])
    assert chunks == []
