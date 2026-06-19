"""R13 (red): agentic seam `ingest.agentic.AgenticOrchestrator` (S15).

Contract §6 Rag_pipeline_architecture.md ("the agentic loop sits as the SAME seam over the same
list of steps — we don't touch the filters"): the agentic orchestrator runs on top of the same
`Sequence[Step]`, the same `StepContext`/`RunDir`/`state.json` as the deterministic one. The only
difference is the DECISION of "which step is next": it is made by an INJECTED `policy(view) -> name|None`
(later backed by the LLM gateway). `default_linear_policy()` is the special case: the first undone
step in order → the agentic loop reproduces the deterministic one.

policy is injected → the test is a pure unit (no network/LLM/DB), on the same fake steps as R5.
RED until S15: the ingest.agentic module does not exist yet.
"""
import pytest


class RecordingStep:
    """The same fake step-contract as in R5: counts runs, writes an artifact."""

    def __init__(self, name, *, params=None, fail=False, order=None):
        self.name = name
        self.artifacts = [f"{name}/out.txt"]
        self._params = dict(params or {})
        self.fail = fail
        self.runs = 0
        self._order = order

    def params(self):
        return self._params

    def run(self, ctx):
        self.runs += 1
        if self._order is not None:
            self._order.append(self.name)
        if self.fail:
            raise RuntimeError(f"{self.name} boom")
        out = ctx.stage_dir / "out.txt"
        out.write_text(f"{self.name}:{self.runs}", encoding="utf-8")
        return str(out)


def _agentic(steps, tmp_path, *, policy, max_iters=None):
    from ingest.agentic import AgenticOrchestrator

    return AgenticOrchestrator(
        steps, base=tmp_path / "runs", policy=policy, max_iters=max_iters
    )


# --- seam: linear policy reproduces the deterministic order ---------------------------------
def test_linear_policy_reproduces_deterministic_order(tmp_path):
    from ingest.agentic import default_linear_policy

    order = []
    steps = [RecordingStep(n, order=order) for n in ("01_a", "02_b", "03_c")]
    summary = _agentic(steps, tmp_path, policy=default_linear_policy()).run("g1")
    assert order == ["01_a", "02_b", "03_c"]      # same order as Orchestrator (R5)
    assert all(s.runs == 1 for s in steps)
    assert set(summary) == {"01_a", "02_b", "03_c"}


# --- the policy decides the next step -------------------------------------------------------
def test_policy_drives_custom_order(tmp_path):
    order = []
    steps = [RecordingStep(n, order=order) for n in ("01_a", "02_b", "03_c")]

    seq = iter(["03_c", "01_a", "02_b", None])  # arbitrary order, then stop
    summary = _agentic(steps, tmp_path, policy=lambda view: next(seq)).run("g1")
    assert order == ["03_c", "01_a", "02_b"]
    assert set(summary) == {"01_a", "02_b", "03_c"}


def test_policy_none_stops_immediately(tmp_path):
    steps = [RecordingStep("01_a")]
    summary = _agentic(steps, tmp_path, policy=lambda view: None).run("g1")
    assert steps[0].runs == 0  # policy immediately None → no step is run
    assert summary == {}


# --- the seam really reuses RunDir/state.json (the same mechanism as R5) --------------------
def test_reuses_rundir_state(tmp_path):
    from ingest.rundir import RunDir

    steps = [RecordingStep("01_a"), RecordingStep("02_b")]
    from ingest.agentic import default_linear_policy

    _agentic(steps, tmp_path, policy=default_linear_policy()).run("g1")
    state = RunDir(tmp_path / "runs", "g1").load_state()["steps"]
    assert state["01_a"]["status"] == "done"
    assert state["02_b"]["artifacts"] == ["02_b/out.txt"]


def test_view_exposes_done_for_policy_and_restart_skips(tmp_path):
    from ingest.agentic import default_linear_policy

    steps = [RecordingStep("01_a"), RecordingStep("02_b")]
    orch = _agentic(steps, tmp_path, policy=default_linear_policy())
    orch.run("g1")
    orch.run("g1")  # restart of the same guid: view.done from state → linear policy returns None
    assert all(s.runs == 1 for s in steps)  # nothing re-executed through the seam


def test_policy_receives_done_flags(tmp_path):
    from ingest.agentic import default_linear_policy

    steps = [RecordingStep("01_a"), RecordingStep("02_b")]
    seen = []

    def spy(view):
        seen.append({v["name"]: v["done"] for v in view})
        return default_linear_policy()(view)

    _agentic(steps, tmp_path, policy=spy).run("g1")
    assert seen[0] == {"01_a": False, "02_b": False}      # start: nothing is done
    assert seen[1] == {"01_a": True, "02_b": False}       # after 01_a → done is visible


# --- safety: a diverging policy is bounded by max_iters -------------------------------------
def test_runaway_policy_is_bounded(tmp_path):
    steps = [RecordingStep("01_a")]
    orch = _agentic(steps, tmp_path, policy=lambda view: "01_a", max_iters=3)
    with pytest.raises(RuntimeError, match="converge"):
        orch.run("g1")
    assert steps[0].runs == 3  # no more than max_iters


# --- step failure: the same contract as the deterministic one ------------------------------
def test_failure_marks_failed_and_raises(tmp_path):
    from ingest.agentic import default_linear_policy
    from ingest.rundir import RunDir

    a = RecordingStep("01_a")
    b = RecordingStep("02_b", fail=True)
    orch = _agentic([a, b], tmp_path, policy=default_linear_policy())
    with pytest.raises(RuntimeError, match="boom"):
        orch.run("g1")
    st = RunDir(tmp_path / "runs", "g1").load_state()["steps"]
    assert st["01_a"]["status"] == "done"
    assert st["02_b"]["status"] == "failed"
