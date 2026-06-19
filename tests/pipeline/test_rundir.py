"""R4 (red): run directory `ingest.rundir.RunDir`.

Pins contract §2/§6 Rag_pipeline_architecture.md:
  • a run is keyed by guid: `runs/<guid>/`, stages are numbered subdirs (pipe files);
  • one `state.json` per run, written ATOMICALLY (temp + os.replace) — a failure doesn't clobber the old one;
  • restart detects done steps: a step is done only if its artifacts are actually on disk
    AND the param hash matched (param changed → invalidation → re-run).

Pure unit, no network/DB. RED before S5: the ingest.rundir module does not exist yet.
"""
import pytest


def _rd(tmp_path, guid="g1"):
    from ingest.rundir import RunDir

    return RunDir(tmp_path / "runs", guid)


# --- directory layout --------------------------------------------------------
def test_root_is_guid_scoped(tmp_path):
    rd = _rd(tmp_path, "abc")
    assert rd.root == tmp_path / "runs" / "abc"


def test_ensure_creates_dir(tmp_path):
    rd = _rd(tmp_path)
    p = rd.ensure()
    assert p.is_dir()


def test_stage_dir_created(tmp_path):
    rd = _rd(tmp_path)
    d = rd.stage_dir("01_extract")
    assert d.is_dir()
    assert d == rd.root / "01_extract"


def test_two_guids_isolated(tmp_path):
    from ingest.rundir import RunDir

    a = RunDir(tmp_path / "runs", "g1")
    b = RunDir(tmp_path / "runs", "g2")
    a.save_state({"who": "a"})
    b.save_state({"who": "b"})
    assert a.load_state()["who"] == "a"
    assert b.load_state()["who"] == "b"


# --- state.json ----------------------------------------------------------------
def test_state_roundtrip(tmp_path):
    rd = _rd(tmp_path)
    rd.save_state({"guid": "g1", "steps": {"a": {"status": "done"}}})
    assert rd.load_state()["steps"]["a"]["status"] == "done"


def test_load_state_default_when_missing(tmp_path):
    rd = _rd(tmp_path)
    assert rd.load_state()["steps"] == {}


def test_save_state_leaves_no_temp(tmp_path):
    rd = _rd(tmp_path)
    rd.save_state({"steps": {}})
    leftovers = [p.name for p in rd.root.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_save_state_atomic_failure_keeps_old(tmp_path, monkeypatch):
    rd = _rd(tmp_path)
    rd.save_state({"v": 1})

    import ingest.rundir as m

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(m.os, "replace", boom)
    with pytest.raises(OSError):
        rd.save_state({"v": 2})
    monkeypatch.undo()

    assert rd.load_state()["v"] == 1  # the old state is intact
    assert not any(p.name.endswith(".tmp") for p in rd.root.iterdir())


# --- step journal / restart ----------------------------------------------------
def test_mark_done_and_is_done(tmp_path):
    rd = _rd(tmp_path)
    rd.stage_dir("01_extract")
    (rd.root / "01_extract" / "full_text.txt").write_text("hi", encoding="utf-8")
    rd.mark_done("01_extract", param_hash="h1", artifacts=["01_extract/full_text.txt"])
    assert rd.is_done("01_extract", param_hash="h1")


def test_is_done_false_when_artifact_missing(tmp_path):
    rd = _rd(tmp_path)
    rd.mark_done("01_extract", param_hash="h1", artifacts=["01_extract/full_text.txt"])
    assert not rd.is_done("01_extract", param_hash="h1")  # no file → re-run


def test_is_done_false_on_param_hash_change(tmp_path):
    rd = _rd(tmp_path)
    rd.stage_dir("03_chunk")
    (rd.root / "03_chunk" / "chunks.jsonl").write_text("{}", encoding="utf-8")
    rd.mark_done("03_chunk", param_hash="h1", artifacts=["03_chunk/chunks.jsonl"])
    assert rd.is_done("03_chunk", param_hash="h1")
    assert not rd.is_done("03_chunk", param_hash="h2")  # param changed → invalidation


def test_mark_failed_records_error(tmp_path):
    rd = _rd(tmp_path)
    rd.mark_failed("02_structure", error="boom")
    st = rd.load_state()["steps"]["02_structure"]
    assert st["status"] == "failed"
    assert "boom" in st["error"]
    assert not rd.is_done("02_structure")
