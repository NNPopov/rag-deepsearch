"""Run directory: `runs/<guid>/` + pipe files + atomic `state.json` (S5).

The run persistence layer (§2/§6 Rag_pipeline_architecture.md). The orchestrator (S6) manages
it; filters don't touch it. Responsibilities:
  • layout `runs/<guid>/NN_stage/...` — the pipes between filters are FILES;
  • one `state.json` per run, updated after each step, written ATOMICALLY (temp + rename),
    so a crash mid-write doesn't corrupt the already-accumulated journal;
  • detection of done steps for restart: a step is `done` only if its artifacts are really on disk
    AND the parameter hash matches (changed a step parameter → invalidation → re-run the step).

The "re-run all steps below" cascade is the orchestrator's responsibility (S6), not this layer's.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

_TMP_SUFFIX = ".state.json.tmp"


class RunDir:
    def __init__(self, base: str | os.PathLike, guid: str) -> None:
        self.guid = guid
        self.root = Path(base) / guid

    # --- directory layout ----------------------------------------------------
    def ensure(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def stage_dir(self, name: str) -> Path:
        d = self.ensure() / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    # --- state.json (atomic) -------------------------------------------------
    def load_state(self) -> dict:
        try:
            raw = self.state_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"guid": self.guid, "steps": {}}
        state = json.loads(raw)
        state.setdefault("steps", {})
        return state

    def save_state(self, state: dict) -> None:
        self.ensure()
        tmp = self.root / _TMP_SUFFIX
        try:
            tmp.write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(tmp, self.state_path)  # atomic swap within the FS
        except BaseException:
            tmp.unlink(missing_ok=True)  # leave no garbage, the old file is untouched
            raise

    # --- step journal ----------------------------------------------------------
    def mark_done(
        self,
        step: str,
        *,
        param_hash: str | None = None,
        artifacts: Sequence[str] = (),
    ) -> None:
        self._update_step(
            step,
            status="done",
            param_hash=param_hash,
            artifacts=list(artifacts),
            error=None,
        )

    def mark_failed(self, step: str, *, error: str) -> None:
        self._update_step(step, status="failed", error=str(error))

    def step_status(self, step: str) -> str | None:
        rec = self.load_state()["steps"].get(step)
        return rec["status"] if rec else None

    def is_done(self, step: str, *, param_hash: str | None = None) -> bool:
        """Whether the step is ready to skip on restart.

        Trust the journal but verify reality: status `done`, artifacts on disk,
        and (if given) a matching parameter hash.
        """
        rec = self.load_state()["steps"].get(step)
        if not rec or rec.get("status") != "done":
            return False
        if param_hash is not None and rec.get("param_hash") != param_hash:
            return False
        return all(self._artifact_exists(a) for a in rec.get("artifacts", []))

    # --- internal ------------------------------------------------------------
    def _update_step(self, step: str, **fields) -> None:
        state = self.load_state()
        rec = state["steps"].get(step, {})
        rec.update(fields)
        rec["updated_at"] = time.time()
        state["steps"][step] = rec
        self.save_state(state)

    def _artifact_exists(self, rel_or_abs: str) -> bool:
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = self.root / p
        return p.exists()
