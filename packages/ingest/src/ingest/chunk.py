"""Chunk filter (S10) — splits leaf sections into chunks.

Single responsibility (§4.3 Rag_pipeline_architecture.md): read `02_structure/sections.jsonl`,
take the LEAVES (`is_leaf`) and split their text into chunks (Chonkie `RecursiveChunker`), and at
overlap>0 add overlap (`OverlapRefinery`, prefix — the tail of the previous chunk into the start
of the next). Output — `03_chunk/chunks.jsonl`, one valid `Chunk` per line (the Chunk→Embed pipe).
Implements the orchestrator step contract (S6).

The `cl100k_base` tokenizer matches the `text-embedding-3-large` embedder, so we COMPUTE
`token_count` OURSELVES via tiktoken (chonkie's internal count differs by a couple of tokens).
torch is NOT needed — it only surfaces in SemanticChunker, which we don't touch.

DI / config-agnostic: size/overlap/tokenizer come via the constructor (composition root, S13).
The chunker itself (`chunk_fn: text -> list[str]`) and the token counter are injected for tests/substitution.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cached_property

from ingest.contracts import Chunk, Section
from ingest.orchestrator import StepContext

ChunkFn = Callable[[str], list[str]]
CountFn = Callable[[str], int]


class ChunkStep:
    name: str = "03_chunk"
    artifacts: Sequence[str] = ("03_chunk/chunks.jsonl",)

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        overlap: int = 64,
        tokenizer: str = "cl100k_base",
        sections_stage: str = "02_structure",
        chunk_fn: ChunkFn | None = None,
        count_fn: CountFn | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._tokenizer = tokenizer
        self._sections_stage = sections_stage  # input: Structure, or Clean (S9) if enabled
        self._chunk_fn = chunk_fn
        self._count_fn = count_fn

    def params(self) -> dict:
        # Restart slice: changed size/overlap/tokenizer/section source → re-chunk.
        return {
            "chunk_size": self._chunk_size,
            "overlap": self._overlap,
            "tokenizer": self._tokenizer,
            "input_stage": self._sections_stage,
        }

    # --- orchestrator step ------------------------------------------------------
    def run(self, ctx: StepContext) -> str:
        src = ctx.rundir.path(self._sections_stage, "sections.jsonl")
        sections = [
            Section.model_validate_json(ln)
            for ln in src.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        chunks = self.chunk_sections(sections)
        out = ctx.stage_dir / "chunks.jsonl"
        out.write_text(
            "\n".join(c.model_dump_json() for c in chunks) + "\n",
            encoding="utf-8",
        )
        return str(out)

    # --- pure chunking (unit-testable, no disk) --------------------------
    def chunk_sections(self, sections: Sequence[Section]) -> list[Chunk]:
        chunk_fn = self._chunk_fn or self._default_chunk_fn
        count_fn = self._count_fn or self._default_count_fn
        chunks: list[Chunk] = []
        for sec in sections:
            if not sec.is_leaf:
                continue  # don't chunk chapter big-blocks — only cut leaves (§4.3)
            pieces = [p for p in chunk_fn(sec.text) if p.strip()]
            for seq, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        external_id=sec.external_id,
                        section_path=sec.path,
                        chapter=int(sec.path.split(".")[0]),
                        heading=sec.heading,
                        seq=seq,
                        text=piece,
                        token_count=count_fn(piece),
                        page_range=sec.page_range,
                        metadata=dict(sec.metadata),
                    )
                )
        return chunks

    # --- default implementations (chonkie + tiktoken) -----------------------------
    @cached_property
    def _default_count_fn(self) -> CountFn:
        import tiktoken

        enc = tiktoken.get_encoding(self._tokenizer)
        return lambda s: len(enc.encode(s))

    @cached_property
    def _default_chunk_fn(self) -> ChunkFn:
        from chonkie import OverlapRefinery, RecursiveChunker

        chunker = RecursiveChunker(tokenizer=self._tokenizer, chunk_size=self._chunk_size)
        refinery = (
            OverlapRefinery(
                tokenizer=self._tokenizer,
                context_size=self._overlap,
                mode="token",
                method="prefix",
            )
            if self._overlap > 0
            else None
        )

        def _chunk(text: str) -> list[str]:
            pieces = list(chunker(text))
            if refinery is not None and pieces:
                pieces = list(refinery(pieces))
            return [p.text for p in pieces]

        return _chunk
