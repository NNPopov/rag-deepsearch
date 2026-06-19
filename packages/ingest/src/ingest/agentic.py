"""Agentic orchestrator (S15) — loop variant on the SAME seam as the deterministic one.

§6 Rag_pipeline_architecture.md: "the agentic loop slots onto the same seam later over the same
step list — we don't touch the filters". The seam = the single step contract
(`run(ctx) -> artifact`) + `RunDir`/`state.json`. Exactly that is reused here; the only
difference from `Orchestrator` is the DECISION of "which step is next".

The deterministic runner walks the list in order. The agentic one asks the INJECTED
`policy(view) -> name | None`, where `view` is the run state (`[{name, done}]` in step order).
The policy is later backed by the LLM gateway (like everything else — via DI, never reading
config itself). `default_linear_policy()` takes the first not-done step → the agentic loop
reproduces the deterministic one (determinism is a special case of the seam).

The policy is AUTHORITATIVE: the orchestrator runs the chosen step (including re-runs); `view.done`
is given only as information for the decision. A diverging policy is bounded by `max_iters`.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence

from ingest.orchestrator import StepContext, Step, _param_hash
from ingest.rundir import RunDir

Policy = Callable[[list[dict]], str | None]


def default_linear_policy() -> Policy:
    """Special-case policy: the first step with `done=False` in order, else None."""

    def policy(view: list[dict]) -> str | None:
        for item in view:
            if not item["done"]:
                return item["name"]
        return None

    return policy


class AgenticOrchestrator:
    def __init__(
        self,
        steps: Sequence[Step],
        *,
        base: str | os.PathLike,
        policy: Policy,
        max_iters: int | None = None,
    ) -> None:
        self._steps = list(steps)
        self._by_name = {s.name: s for s in self._steps}
        self._base = base
        self._policy = policy
        # headroom for non-linear policies (revisits); +1 for the linear policy's final None
        self._max_iters = max_iters if max_iters is not None else len(self._steps) * 4 + 1

    @property
    def steps(self) -> tuple[Step, ...]:
        return tuple(self._steps)

    @property
    def base(self) -> str | os.PathLike:
        return self._base

    def run(self, guid: str) -> dict:
        rd = RunDir(self._base, guid)
        rd.ensure()
        hashes = {s.name: _param_hash(s.params()) for s in self._steps}
        summary: dict = {}
        for _ in range(self._max_iters):
            view = [
                {"name": s.name, "done": rd.is_done(s.name, param_hash=hashes[s.name])}
                for s in self._steps
            ]
            choice = self._policy(view)
            if choice is None:
                return summary
            step = self._by_name[choice]
            ctx = StepContext(rundir=rd, guid=guid, stage_dir=rd.stage_dir(step.name))
            try:
                artifact = step.run(ctx)
            except BaseException as exc:
                rd.mark_failed(step.name, error=str(exc))
                raise
            rd.mark_done(
                step.name, param_hash=hashes[step.name], artifacts=list(step.artifacts)
            )
            summary[step.name] = artifact
        raise RuntimeError(
            f"agentic policy did not converge in {self._max_iters} iterations"
        )
