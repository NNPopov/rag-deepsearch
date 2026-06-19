"""Deterministic run orchestrator (S6).

A thin, declarative runner over RunDir (§6 Rag_pipeline_architecture.md):
a step list + one shared loop, no business logic. Steps do NOT call each other — the orchestrator
holds the order, the work on `runs/<guid>/`, restarts and errors.

Step contract (Protocol `Step`):
  • `name`        — step name = stage-folder name (e.g. "01_extract");
  • `params()`    — the resolved slice of step settings (dict); its hash lives in state.json;
  • `artifacts`   — the step's output paths relative to the run root (for readiness detection);
  • `run(ctx)`    — does the work, returns an artifact. `ctx.stage_dir` is its own folder.

Step loop:
  • param_hash = hash of `params()`; if the step `is_done` with the same hash and NOT forced — skip;
  • otherwise run → `mark_done`; any executed step FORCES all steps below (cascade: their input
    changed). Changing a step's parameter → its hash mismatches → it and everything below re-run,
    steps above with unchanged parameters stay skipped;
  • a step exception → `mark_failed` + re-raise; steps below don't run.

The same step list is later reused by the agentic loop (`orchestrators/agentic.py`) — one seam.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ingest.rundir import RunDir


@dataclass
class StepContext:
    """What the orchestrator hands the step. The step finds its input by convention via `rundir.path(...)`."""

    rundir: RunDir
    guid: str
    stage_dir: Path


@runtime_checkable
class Step(Protocol):
    name: str
    artifacts: Sequence[str]

    def params(self) -> dict: ...

    def run(self, ctx: StepContext) -> Any: ...


def _param_hash(params: dict) -> str:
    """Stable hash of the step's resolved parameter slice (for restart invalidation)."""
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Orchestrator:
    def __init__(self, steps: Sequence[Step], *, base: str | os.PathLike) -> None:
        self._steps = list(steps)
        self._base = base

    @property
    def steps(self) -> tuple[Step, ...]:
        """Run steps in order (read-only — for composition-root introspection)."""
        return tuple(self._steps)

    @property
    def base(self) -> str | os.PathLike:
        """Root of the runs directory (`runs/`), where the orchestrator places `<guid>/`."""
        return self._base

    def run(self, guid: str) -> dict:
        rd = RunDir(self._base, guid)
        rd.ensure()
        summary: dict[str, Any] = {}
        force = False  # ran a step → re-run all below (their input changed)
        for step in self._steps:
            h = _param_hash(step.params())
            if not force and rd.is_done(step.name, param_hash=h):
                summary[step.name] = "skipped"
                continue
            ctx = StepContext(rundir=rd, guid=guid, stage_dir=rd.stage_dir(step.name))
            try:
                artifact = step.run(ctx)
            except BaseException as exc:
                rd.mark_failed(step.name, error=str(exc))
                raise
            rd.mark_done(step.name, param_hash=h, artifacts=list(step.artifacts))
            summary[step.name] = artifact
            force = True
        return summary
