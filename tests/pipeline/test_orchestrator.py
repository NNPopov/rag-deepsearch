"""R5 (red): deterministic orchestrator `ingest.orchestrator.Orchestrator`.

Pins contract §6 Rag_pipeline_architecture.md:
  • a single step interface `run(ctx) -> artifact`; a step declares its name, params, artifacts;
  • the orchestrator is thin/declarative: a list of steps + a common loop, owns `runs/<guid>/`
    via RunDir, steps do NOT call each other;
  • restart by guid: done steps (is_done) are skipped — their run() is not called again;
  • invalidation by param hash: a step's param changed → that step is re-run
    AND ALL BELOW IT (cascade); steps above with unchanged params stay skipped;
  • a step error → mark_failed in state.json + re-raise; steps below do not run.

Pure unit, no network/DB. RED before S6: the ingest.orchestrator module does not exist yet.
"""
import pytest


class RecordingStep:
    """Fake step contract: counts runs, writes an artifact into its own stage dir."""

    def __init__(self, name, *, params=None, fail=False, order=None):
        self.name = name
        self.artifacts = [f"{name}/out.txt"]
        self._params = dict(params or {})
        self.fail = fail
        self.runs = 0
        self._order = order

    def params(self):
        return self._params

    def set_param(self, key, value):
        self._params[key] = value

    def run(self, ctx):
        self.runs += 1
        if self._order is not None:
            self._order.append(self.name)
        if self.fail:
            raise RuntimeError(f"{self.name} boom")
        out = ctx.stage_dir / "out.txt"
        out.write_text(f"{self.name}:{self.runs}", encoding="utf-8")
        return str(out)


def _orch(steps, tmp_path):
    from ingest.orchestrator import Orchestrator

    return Orchestrator(steps, base=tmp_path / "runs")


# --- order and artifacts -------------------------------------------------------
def test_runs_steps_in_declared_order(tmp_path):
    order = []
    steps = [RecordingStep(n, order=order) for n in ("01_a", "02_b", "03_c")]
    _orch(steps, tmp_path).run("g1")
    assert order == ["01_a", "02_b", "03_c"]
    assert all(s.runs == 1 for s in steps)


def test_creates_stage_dirs_and_artifacts(tmp_path):
    steps = [RecordingStep("01_a"), RecordingStep("02_b")]
    _orch(steps, tmp_path).run("g1")
    root = tmp_path / "runs" / "g1"
    assert (root / "01_a" / "out.txt").read_text(encoding="utf-8") == "01_a:1"
    assert (root / "02_b" / "out.txt").read_text(encoding="utf-8") == "02_b:1"


def test_records_done_in_state(tmp_path):
    from ingest.rundir import RunDir

    steps = [RecordingStep("01_a")]
    _orch(steps, tmp_path).run("g1")
    state = RunDir(tmp_path / "runs", "g1").load_state()
    assert state["steps"]["01_a"]["status"] == "done"
    assert state["steps"]["01_a"]["artifacts"] == ["01_a/out.txt"]


# --- restart by guid -----------------------------------------------------------
def test_restart_skips_done_steps(tmp_path):
    steps = [RecordingStep("01_a"), RecordingStep("02_b")]
    orch = _orch(steps, tmp_path)
    orch.run("g1")
    orch.run("g1")  # second run of the same guid
    assert all(s.runs == 1 for s in steps)  # nothing was re-run


def test_restart_reruns_when_artifact_deleted(tmp_path):
    steps = [RecordingStep("01_a")]
    orch = _orch(steps, tmp_path)
    orch.run("g1")
    (tmp_path / "runs" / "g1" / "01_a" / "out.txt").unlink()  # artifact gone
    orch.run("g1")
    assert steps[0].runs == 2  # no file on disk → re-run


# --- invalidation cascade by param hash -------------------------------------
def test_param_change_reruns_step_and_all_below(tmp_path):
    a = RecordingStep("01_a")
    b = RecordingStep("02_b", params={"chunk_size": 800})
    c = RecordingStep("03_c")
    orch = _orch([a, b, c], tmp_path)
    orch.run("g1")
    assert (a.runs, b.runs, c.runs) == (1, 1, 1)

    b.set_param("chunk_size", 1000)  # changed the middle step's param
    orch.run("g1")
    assert a.runs == 1  # above and unchanged → skipped
    assert b.runs == 2  # the step itself re-run
    assert c.runs == 2  # and all below — cascade (input changed)


def test_unchanged_run_is_fully_skipped(tmp_path):
    a = RecordingStep("01_a", params={"x": 1})
    b = RecordingStep("02_b", params={"y": 2})
    orch = _orch([a, b], tmp_path)
    orch.run("g1")
    orch.run("g1")
    assert (a.runs, b.runs) == (1, 1)


# --- errors --------------------------------------------------------------------
def test_failure_marks_failed_and_raises(tmp_path):
    from ingest.rundir import RunDir

    a = RecordingStep("01_a")
    b = RecordingStep("02_b", fail=True)
    c = RecordingStep("03_c")
    orch = _orch([a, b, c], tmp_path)
    with pytest.raises(RuntimeError):
        orch.run("g1")
    assert c.runs == 0  # the step below the failed one does not run
    st = RunDir(tmp_path / "runs", "g1").load_state()["steps"]
    assert st["01_a"]["status"] == "done"
    assert st["02_b"]["status"] == "failed"
    assert "boom" in st["02_b"]["error"]


def test_resume_after_failure_skips_done_prefix(tmp_path):
    a = RecordingStep("01_a")
    b = RecordingStep("02_b", fail=True)
    orch = _orch([a, b], tmp_path)
    with pytest.raises(RuntimeError):
        orch.run("g1")
    b.fail = False  # "fixed" the step
    orch.run("g1")
    assert a.runs == 1  # the already-done prefix is left untouched
    assert b.runs == 2  # the re-run step carried through to done
