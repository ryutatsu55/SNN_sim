"""新しい基準ネットワークの科学的条件と実行管理を検証する。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.experiments.lif_network import (
    NetworkSettings, app_config, frozen_spikes, inventory, make_plan, preset_settings, save_json,
)
from src.utils.experiments import lif_network_job as job


@pytest.mark.parametrize("patch", [
    {"n_exc": 1000}, {"seeds": [0]}, {"seeds": [11, 11]},
    {"gmax_values": [float("nan")]}, {"beta_inh_values": [2]},
    {"dt_ms": 0.3}, {"frozen_ms": 11001}, {"rest_rates_hz": [0]},
])
def test_invalid_conditions_rejected(patch):
    with pytest.raises(ValidationError):
        NetworkSettings.model_validate({**NetworkSettings().model_dump(), **patch})


def test_plan_preserves_main_network_and_paired_wiring():
    cfg = NetworkSettings()
    plan = make_plan(cfg)
    assert len(plan) == 27
    assert len({r["case_id"] for r in plan}) == 27
    smoke = preset_settings(cfg, "smoke")
    assert (smoke.n, smoke.n_exc, smoke.probability) == (1000, 800, 0.1)
    assert len(make_plan(smoke)) == 1
    for row in plan:
        app = app_config(cfg, row)
        assert app.simulation.seed == row["seed"]
        assert app.network.connection.p == 0.1
        assert not app.network.connection.allow_self_connections
        assert app.neurons["Exc"].type == "akita_escape_lif"
        assert app.synapses["Inh"].synapse.E == -80
        assert app.synapses["Inh"].plasticity.pairing == "nearest"


def test_frozen_chunks_form_one_continuous_observation(tmp_path):
    cfg = preset_settings(NetworkSettings(), "smoke")
    (tmp_path / "chunks").mkdir()
    for i, (offset, local, phase) in enumerate([
        (0, [20], "development"), (1000, [499.9], "frozen"), (1500, [0.1], "frozen"),
    ]):
        np.savez_compressed(tmp_path / "chunks" / f"{i:06d}.npz", times_ms=local,
                            ids=[0], phase=phase, start_ms=offset)
    times, ids = frozen_spikes(tmp_path, cfg)
    np.testing.assert_allclose(times, [499.9, 500.1])
    assert len(ids) == 2


def test_resume_rejects_changed_plan_and_source(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("pass\n")
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setattr(job, "source_paths", lambda: [source])
    out = job.prepare(tmp_path / "run", preset_settings(NetworkSettings(), "smoke"))
    job.check_sources(out)
    source.write_text("x = 1\n")
    with pytest.raises(ValueError, match="ソース"):
        job.check_sources(out)
    source.write_text("pass\n")
    (out / "settings.json").write_text("{}")
    with pytest.raises(ValueError, match="設定"):
        job.check_sources(out)


def test_resume_requires_complete_unchanged_artifacts(tmp_path):
    names = ["result.json", "frozen_metrics.json", "activity.png", "connectivity.npz",
             "weights/frozen.npz", "metrics.csv"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("test")
    save_json(tmp_path / "result.json", dict(status="complete"))
    save_json(tmp_path / "artifacts.json", inventory(tmp_path, exclude=("artifacts.json",)))
    assert job.verify_attempt(tmp_path)
    (tmp_path / "weights/frozen.npz").write_text("changed")
    assert not job.verify_attempt(tmp_path)


@pytest.mark.parametrize("failure", ["body", "final", "receipt", None])
def test_upload_separates_failure_and_preserves_local_results(tmp_path, monkeypatch, failure):
    marker = tmp_path / "spikes.npz"
    marker.write_bytes(b"local data")
    calls = []
    monkeypatch.setattr(job, "check_remote", lambda remote: None)

    def execute(command, log, timeout, **kwargs):
        phase = ("body", "final", "receipt")[len(calls)]
        calls.append(command)
        return dict(status="failed" if phase == failure else "complete", returncode=1 if phase == failure else 0)

    monkeypatch.setattr(job, "execute", execute)
    state = job.upload(tmp_path, "remote:folder with spaces")
    assert state["status"] == ("complete" if failure is None else "failed")
    assert marker.read_bytes() == b"local data"
    assert all(command[1] in ("copy", "copyto") for command in calls)
    assert calls[0][3] == "remote:folder with spaces/" + tmp_path.name


def test_timeout_collects_child_process_group(tmp_path):
    child_file = tmp_path / "child.pid"
    code = ("import subprocess,sys,time; from pathlib import Path; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)")
    result = job.execute([sys.executable, "-c", code, str(child_file)], tmp_path / "log", 0.3)
    assert result["status"] == "timeout"
    assert result["returncode"] < 0
    deadline = time.monotonic() + 3
    while job.process_identity(int(child_file.read_text())) is not None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert job.process_identity(int(child_file.read_text())) is None


def test_resume_recovers_orphan_worker_without_guessing_exit_code(tmp_path):
    case_dir = tmp_path / "cases/case_0000"
    case_dir.mkdir(parents=True)
    log = case_dir / "attempt_001.log"
    log.touch()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        save_json(log.with_suffix(".process.json"), dict(pid=process.pid, process_start=job.process_identity(process.pid)))
        job.recover_interrupted_attempts(tmp_path)
        process.wait(timeout=3)
        record = job.read_json(log.with_suffix(".exit.json"))
        assert record["status"] == "interrupted"
        assert record["returncode"] is None
    finally:
        if process.poll() is None:
            job.stop_group(process)


def wait_status(out, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = job.read_json(out / "status.json", {})
        if status.get("status") in ("complete", "failed", "paused") and not job.running(out):
            return status
        time.sleep(0.1)
    raise AssertionError(f"終了しませんでした: {out}")


@pytest.mark.parametrize("dt_ms", [0.1, 0.05])
def test_background_survives_launcher_and_freezes_learning(tmp_path, dt_ms):
    """ランチャ終了後も管理とGeNNが完走し、再実行でも発火列が一致する。"""
    cfg = preset_settings(NetworkSettings(dt_ms=dt_ms), "smoke")
    outputs = []
    for label in ("first", "replay"):
        out = job.prepare(tmp_path / label, cfg)
        launcher = subprocess.run([sys.executable, str(job.CLI), "resume", "--run-dir", str(out)],
                                  capture_output=True, text=True, check=True)
        record = json.loads(launcher.stdout)
        assert record["pid"] != os.getpid()
        status = wait_status(out)
        assert status["status"] == "complete", (out / "run.log").read_text()
        identity = job.read_json(out / "job.json")
        assert identity["session_id"] == identity["pid"]
        outcome = job.case_outcome(out, "case_0000")
        attempt = out / outcome["attempt"]
        assert job.verify_attempt(attempt)
        with np.load(attempt / "connectivity.npz") as wiring:
            assert len(wiring["row"]) > 90000
            assert not np.any(wiring["row"] == wiring["col"])
            ee = (wiring["row"] < cfg.n_exc) & (wiring["col"] < cfg.n_exc)
            np.testing.assert_allclose(wiring["delays_ms"][ee], 1.5)
            np.testing.assert_allclose(wiring["delays_ms"][~ee], 0.8)
        with np.load(attempt / "weights/frozen.npz") as data:
            assert np.any(data["w"] > 0), "STDPによる発達がありません"
            frozen = data["w"].copy()
        with np.load(attempt / "weights/000004.npz") as data:
            np.testing.assert_array_equal(data["w"], frozen)
        before, after = np.load(attempt / "freeze_state.npz"), np.load(attempt / "final_state.npz")
        try:
            assert any(not np.array_equal(before[key], after[key]) for key in before.files if key.endswith("__x")), "固定後のSTPが動いていません"
        finally:
            before.close()
            after.close()
        outputs.append(attempt)
    for chunk in range(4):
        with np.load(outputs[0] / "chunks" / f"{chunk:06d}.npz") as first, np.load(outputs[1] / "chunks" / f"{chunk:06d}.npz") as second:
            np.testing.assert_array_equal(first["times_ms"], second["times_ms"])
            np.testing.assert_array_equal(first["ids"], second["ids"])


def test_stop_resume_and_duplicate_manager_protection(tmp_path, monkeypatch):
    """管理のみを模擬し、長い科学計算を追加しない。"""
    cfg = preset_settings(NetworkSettings(), "smoke")
    cfg = NetworkSettings.model_validate({**cfg.model_dump(), "seeds": [11, 22]})
    out = job.prepare(tmp_path / "run", cfg)
    attempts = []
    uploads = []
    monkeypatch.setattr(job, "verify_attempt", lambda path: True)
    save_json(out / "job_options.json", dict(remote="remote:results", upload_timeout_s=10))

    def upload(*args):
        assert job.case_outcome(out, "case_0000")["status"] == "complete"
        assert job.case_outcome(out, "case_0001")["status"] == "complete"
        uploads.append(args)
        return dict(status="complete")

    monkeypatch.setattr(job, "upload", upload)

    def execute(command, log, timeout, forced, heartbeat):
        attempt = Path(command[-1])
        attempt.mkdir()
        Path(log).touch()
        attempts.append(attempt)
        if len(attempts) == 1:
            (out / "STOP").touch()
        return dict(status="complete", returncode=0, elapsed_s=0.01)

    monkeypatch.setattr(job, "execute", execute)
    old_term, old_int = signal_handlers()
    try:
        assert job.manage(out) == 1
        assert job.read_json(out / "status.json")["status"] == "paused"
        assert not uploads
        assert job.manage(out) == 0
        assert len(attempts) == 2  # 最初の成功条件を再計算しない。
        assert len(uploads) == 1
        with (out / ".lock").open("a") as lock:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(RuntimeError, match="既に"):
                job.manage(out)
    finally:
        import signal
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)


def signal_handlers():
    import signal
    return signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)


def test_case_failure_continues_and_is_uploaded_as_failure(tmp_path, monkeypatch):
    cfg = preset_settings(NetworkSettings(), "smoke")
    cfg = NetworkSettings.model_validate({**cfg.model_dump(), "seeds": [11, 22]})
    out = job.prepare(tmp_path / "run", cfg)
    save_json(out / "job_options.json", dict(remote="remote:results", upload_timeout_s=10))
    calls, uploads = [], []
    monkeypatch.setattr(job, "verify_attempt", lambda path: True)
    monkeypatch.setattr(job, "plot_campaign", lambda path: None)

    def execute(command, log, timeout, forced, heartbeat):
        Path(command[-1]).mkdir()
        Path(log).touch()
        calls.append(command)
        return dict(status="failed" if len(calls) == 1 else "complete",
                    returncode=-11 if len(calls) == 1 else 0, elapsed_s=0.01)

    def upload(*args):
        uploads.append(args)
        assert len(calls) == 2
        return dict(status="complete")

    monkeypatch.setattr(job, "execute", execute)
    monkeypatch.setattr(job, "upload", upload)
    old_term, old_int = signal_handlers()
    try:
        assert job.manage(out) == 1
        status = job.read_json(out / "status.json")
        assert status["status"] == "failed"
        assert status["counts"]["failed"] == status["counts"]["complete"] == 1
        assert status["upload_status"] == "complete"
        assert len(uploads) == 1
        assert job.case_outcome(out, "case_0000")["returncode"] == -11
    finally:
        import signal
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
