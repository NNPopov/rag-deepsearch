"""End-to-end LLM gateway over LiteLLM — the single point for `completion` + `embedding`.

Contract (§5 Rag_pipeline_architecture.md, CLAUDE.md §2):
  • config-agnostic: model_map / dimension / retries come ONLY via the constructor;
    the gateway itself doesn't read env/Dynaconf (the app's composition root provides the values);
  • task → model mapping via model_map[task] — the code doesn't know provider names;
  • embedding(): a vector of length `dimension`, response order = input order (sorted by index);
  • retries of transient errors (total attempts = retries + 1);
  • cost logging — the optional cost_logger hook.

The underlying litellm calls are injected (`completion_fn`/`embedding_fn`) — this gives a
network-free mock in tests; in prod it defaults to `litellm.completion`/`litellm.embedding`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence


class Gateway:
    def __init__(
        self,
        *,
        model_map: dict[str, str],
        dimension: int,
        retries: int = 2,
        completion_fn: Callable | None = None,
        embedding_fn: Callable | None = None,
        cost_logger: Callable | None = None,
    ) -> None:
        self._model_map = dict(model_map)
        self._dimension = int(dimension)
        self._retries = int(retries)
        self._completion_fn = completion_fn
        self._embedding_fn = embedding_fn
        self._cost_logger = cost_logger

    # --- public interface ---------------------------------------------------
    def completion(self, *, task: str, messages: list[dict], **kw) -> str:
        model = self._resolve(task)
        resp = self._call(self._completion, model=model, messages=messages, **kw)
        self._log_cost(model=model, response=resp)
        return resp.choices[0].message.content

    def completion_stream(self, *, task: str, messages: list[dict], **kw):
        """Streaming completion: yields text deltas (for the SYNTH SSE stream).

        Over `litellm.completion(stream=True)` — the provider returns an iterator of chunks,
        the delta content lives in `chunk.choices[0].delta.content` (service chunks with no
        content — None/"" — are skipped). We do NOT apply retries here: a stream can't be
        replayed from the middle, and the initiating call doesn't do network work yet.
        The generator is lazy — an unknown task (KeyError from `_resolve`) surfaces at the
        first iteration.
        """
        model = self._resolve(task)
        resp = self._completion(model=model, messages=messages, stream=True, **kw)
        for chunk in resp:
            choices = getattr(chunk, "choices", None) or (
                chunk.get("choices") if isinstance(chunk, dict) else None
            )
            if not choices:
                continue
            first = choices[0]
            delta = getattr(first, "delta", None)
            if delta is None and isinstance(first, dict):
                delta = first.get("delta")
            content = getattr(delta, "content", None)
            if content is None and isinstance(delta, dict):
                content = delta.get("content")
            if content:
                yield content

    def embedding(self, *, texts: Sequence[str], task: str = "embed") -> list[list[float]]:
        model = self._resolve(task)
        resp = self._call(
            self._embedding, model=model, input=list(texts), dimensions=self._dimension
        )
        vectors = self._ordered_vectors(resp, expected=len(texts))
        for v in vectors:
            if len(v) != self._dimension:
                raise ValueError(
                    f"embedding length {len(v)} != dimension {self._dimension} "
                    f"(model={model})"
                )
        return vectors

    # --- internal ------------------------------------------------------------
    def _resolve(self, task: str) -> str:
        return self._model_map[task]  # unknown task → KeyError (contract R3)

    def _call(self, fn: Callable, **kw):
        """Call with retries of transient errors; total attempts = retries + 1."""
        last: Exception | None = None
        for _ in range(self._retries + 1):
            try:
                return fn(**kw)
            except Exception as exc:  # noqa: BLE001 — the gateway retries any transient failures
                last = exc
        raise last  # type: ignore[misc]

    @staticmethod
    def _ordered_vectors(resp, *, expected: int) -> list[list[float]]:
        """Sort data by index → response order == input order."""
        items = list(resp.data)
        items.sort(key=lambda d: d["index"] if isinstance(d, dict) else d.index)
        vectors = [d["embedding"] if isinstance(d, dict) else d.embedding for d in items]
        if len(vectors) != expected:
            raise ValueError(
                f"embedding count {len(vectors)} != input count {expected}"
            )
        return vectors

    def _log_cost(self, *, model: str, response) -> None:
        if self._cost_logger is None:
            return
        cost = None
        hidden = getattr(response, "_hidden_params", None)
        if isinstance(hidden, dict):
            cost = hidden.get("response_cost")
        self._cost_logger(model=model, cost=cost)

    # --- lazy defaults on litellm --------------------------------------------
    @property
    def _completion(self) -> Callable:
        if self._completion_fn is not None:
            return self._completion_fn
        import litellm  # noqa: PLC0415 — pull the provider only in prod

        return litellm.completion

    @property
    def _embedding(self) -> Callable:
        if self._embedding_fn is not None:
            return self._embedding_fn
        import litellm  # noqa: PLC0415

        return litellm.embedding
