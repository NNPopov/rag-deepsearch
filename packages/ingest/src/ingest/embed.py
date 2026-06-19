"""Embed filter (S11) — vectorizes chunks via the single LLM gateway.

Single responsibility (§4.4 Rag_pipeline_architecture.md): read `03_chunk/chunks.jsonl`,
run the chunk texts through `llm_gateway` (LiteLLM `embedding`, `text-embedding-3-large`,
`dimensions=1536`) and write `04_embed/chunks_embedded.jsonl` — one `EmbeddedChunk`
per line (the Embed→Persist pipe). Implements the orchestrator step contract (S6).

We batch by token budget: `Chunk.token_count` is already computed at S10 with the same tokenizer
(cl100k) as the embedder, so we don't recompute. Output order = input order (the gateway
sorts the response by index). A chunk larger than the budget isn't lost — it goes in its own batch.

DI / config-agnostic: the gateway itself and the model name come via the constructor (composition root, S13).
The vector length (`db.DIMENSION`) is guaranteed by the gateway and validated by the `EmbeddedChunk` contract.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

from ingest.contracts import Chunk, EmbeddedChunk
from ingest.orchestrator import StepContext


class EmbedStep:
    name: str = "04_embed"
    artifacts: Sequence[str] = ("04_embed/chunks_embedded.jsonl",)

    def __init__(
        self,
        *,
        gateway,
        embedding_model: str,
        max_batch_tokens: int = 100_000,
        max_batch_size: int = 2048,
        task: str = "embed",
    ) -> None:
        self._gateway = gateway
        self._embedding_model = embedding_model
        self._max_batch_tokens = max_batch_tokens
        self._max_batch_size = max_batch_size
        self._task = task

    def params(self) -> dict:
        # Restart slice: changed model/budget/limit/task → hash mismatch → re-embed.
        return {
            "embedding_model": self._embedding_model,
            "max_batch_tokens": self._max_batch_tokens,
            "max_batch_size": self._max_batch_size,
            "task": self._task,
        }

    # --- orchestrator step ------------------------------------------------------
    def run(self, ctx: StepContext) -> str:
        src = ctx.rundir.path("03_chunk", "chunks.jsonl")
        chunks = [
            Chunk.model_validate_json(ln)
            for ln in src.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        embedded = self.embed_chunks(chunks)
        out = ctx.stage_dir / "chunks_embedded.jsonl"
        out.write_text(
            "\n".join(ec.model_dump_json() for ec in embedded) + "\n",
            encoding="utf-8",
        )
        return str(out)

    # --- pure vectorization (unit-testable, gateway injected) -------------
    def embed_chunks(self, chunks: Sequence[Chunk]) -> list[EmbeddedChunk]:
        out: list[EmbeddedChunk] = []
        for batch in self._batches(chunks):
            vectors = self._gateway.embedding(
                texts=[c.text for c in batch], task=self._task
            )
            for c, vector in zip(batch, vectors):
                data = c.model_dump(exclude={"id", "content_hash"})  # computed → not into the constructor
                out.append(
                    EmbeddedChunk(
                        **data,
                        vector=vector,
                        embedding_model=self._embedding_model,
                    )
                )
        return out

    # --- batching by token budget and size limit -----------------
    def _batches(self, chunks: Sequence[Chunk]) -> Iterator[list[Chunk]]:
        batch: list[Chunk] = []
        tokens = 0
        for c in chunks:
            # Cut BEFORE appending if a non-empty batch would overflow; a single
            # chunk larger than the budget stays in its own batch (not lost, not split).
            if batch and (
                tokens + c.token_count > self._max_batch_tokens
                or len(batch) >= self._max_batch_size
            ):
                yield batch
                batch, tokens = [], 0
            batch.append(c)
            tokens += c.token_count
        if batch:
            yield batch
