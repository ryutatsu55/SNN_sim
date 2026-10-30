"""LIF実験の切り離し実行、条件単位の再開、終了後の結果転送。"""

import csv
from datetime import datetime
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from src.utils.experiments.lif_network import (
    ROOT, NetworkSettings, inventory, make_plan, save_json, sha256,
)

CLI = ROOT / "scripts/lif_network.py"


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def process_identity(pid):
    """PID再利用とゾンビを区別する。Linux/WSL用。"""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if stat[0] == "Z" else stat[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def running(out):
    job = read_json(Path(out) / "job.json", {})
    identity = process_identity(job.get("pid", -1))
    return identity is not None and identity == job.get("process_start")


def source_paths():
    return sorted([*ROOT.glob("src/**/*.py"), CLI, ROOT / "configs/lif_network.yaml",
                   ROOT / "requirements.txt", ROOT / "scripts/akita_soc/analysis/metrics.py"])


def check_sources(out):
    manifest = read_json(Path(out) / "manifest.json")
    expected = manifest["sources"]
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"実行時からソースが変わっています。新規実行にしてください: {relative}")
    for name, digest in manifest["inputs"].items():
        if sha256(Path(out) / name) != digest:
            raise ValueError(f"保存済みの計画・設定が変更されています: {name}")


def check_remote(remote):
    """認証内容を表示せず、既存remote名だけを確認する。"""
    if not remote:
        return
    if ":" not in remote or remote.startswith(("/", ":", "-")):
        raise ValueError("保存先は remote:フォルダ の形式にしてください")
    if shutil.which("rclone") is None:
        raise ValueError("rcloneが見つかりません")
    completed = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True, timeout=30, check=True)
    if remote.split(":", 1)[0] + ":" not in completed.stdout.splitlines():
        raise ValueError("指定したremoteがrcloneに登録されていません")


def prepare(out, cfg, remote=None):
    out = Path(out).resolve()
    check_remote(remote)
    out.mkdir(parents=True, exist_ok=False)
    (out / "cases").mkdir()
    sources = {}
    for path in source_paths():
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[str(relative)] = sha256(path)
    packages = {}
    for name in ("numpy", "scipy", "pydantic", "matplotlib", "pygenn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    save_json(out / "settings.json", cfg.model_dump())
    save_json(out / "plan.json", make_plan(cfg))
    save_json(out / "manifest.json", dict(created=now(), sources=sources, python=sys.version,
              executable=sys.executable, packages=packages, git_commit=git.stdout.strip(),
              inputs={name: sha256(out / name) for name in ("settings.json", "plan.json")},
              resume_scope="条件単位。中断した条件の内部状態は復元しない。"))
    save_json(out / "job_options.json", dict(remote=remote, upload_timeout_s=7200))
    save_json(out / "status.json", dict(status="prepared", planned=len(make_plan(cfg)), updated=now()))
    return out


def launch(out, mode="run", remote=None):
    out = Path(out).resolve()
    if running(out):
        raise RuntimeError("この実行は既に起動しています")
    if mode == "run":
        check_sources(out)
    if remote is not None:
        check_remote(remote)
    command = [sys.executable, "-u", str(CLI), "_manage", "--run-dir", str(out), "--mode", mode]
    if remote is not None:
        command += ["--remote", remote]
    with (out / "run.log").open("ab") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    # 起動確認だけ待つ。以降は端末と無関係に管理プロセスが生存する。
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = read_json(out / "job.json", {})
        if job.get("pid") == process.pid and running(out):
            return dict(run_dir=str(out), pid=process.pid, mode=mode)
        if process.poll() is not None:
            raise RuntimeError(f"管理プロセスの起動に失敗しました: {out / 'run.log'}")
        time.sleep(0.05)
    raise RuntimeError(f"起動確認が時間内に終わりませんでした。statusで確認してください: {out}")


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 2
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    # 親が終了していても、同一グループ内のコンパイラ等を回収する。
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(command, log_path, timeout_s, forced=lambda: False, heartbeat=lambda: None):
    started = time.monotonic()
    with Path(log_path).open("ab") as log:
        process = subprocess.Popen(list(map(str, command)), cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        save_json(Path(log_path).with_suffix(".process.json"), dict(
            pid=process.pid, process_start=process_identity(process.pid),
            command=list(map(str, command)), started=now()))
        reason = None
        try:
            while process.poll() is None:
                heartbeat()
                if forced():
                    reason = "interrupted"
                elif time.monotonic() - started > timeout_s:
                    reason = "timeout"
                if reason:
                    stop_group(process)
                    break
                time.sleep(0.2)
        except BaseException:
            stop_group(process)
            raise
    return dict(returncode=process.returncode, status=reason or ("complete" if process.returncode == 0 else "failed"),
                elapsed_s=time.monotonic() - started)


def recover_interrupted_attempts(out):
    """管理プロセスが強制終了した場合の残存ワーカーを識別し、履歴を補う。"""
    for log in (Path(out) / "cases").glob("*/attempt_*.log"):
        exit_path = log.with_suffix(".exit.json")
        if exit_path.exists():
            continue
        process = read_json(log.with_suffix(".process.json"), {})
        pid = process.get("pid", -1)
        identity = process_identity(pid)
        if identity is not None and identity == process.get("process_start"):
            if os.getpgid(pid) != pid:
                raise RuntimeError("前回ワーカーのプロセスグループが想定と違うため再開を停止しました")
            try:
                os.killpg(pid, signal.SIGTERM)
                deadline = time.monotonic() + 2
                while process_identity(pid) is not None and time.monotonic() < deadline:
                    time.sleep(0.05)
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        save_json(exit_path, dict(status="interrupted", returncode=None, ended=now(),
                  error="前回の管理処理が終了コードの保存前に停止しました。終了コードは不明です。"))


def verify_attempt(attempt):
    attempt = Path(attempt)
    result = read_json(attempt / "result.json", {})
    artifacts = read_json(attempt / "artifacts.json", {})
    required = {"result.json", "frozen_metrics.json", "activity.png", "connectivity.npz", "weights/frozen.npz", "metrics.csv"}
    if result.get("status") != "complete" or not required <= artifacts.keys():
        return False
    for relative, info in artifacts.items():
        path = attempt / relative
        if not path.is_file() or path.stat().st_size != info["size"] or sha256(path) != info["sha256"]:
            return False
    return True


def case_outcome(out, case_id):
    return read_json(Path(out) / "cases" / case_id / "outcome.json", {})


def summarize(out):
    out = Path(out)
    cfg = NetworkSettings.model_validate(read_json(out / "settings.json"))
    rows = []
    for case in read_json(out / "plan.json"):
        outcome = case_outcome(out, case["case_id"])
        row = {**case, "status": outcome.get("status", "not_run"), "attempt": outcome.get("attempt"),
               "elapsed_s": outcome.get("elapsed_s"), "returncode": outcome.get("returncode")}
        if row["status"] == "complete":
            metrics = read_json(out / row["attempt"] / "frozen_metrics.json", {})
            row.update({k: metrics.get(k) for k in ("mean_rate_hz", "participants", "llr", "delta_cr", "burstiness_index")})
        rows.append(row)
    names = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    counts = {state: sum(row["status"] == state for row in rows)
              for state in ("complete", "failed", "timeout", "interrupted", "not_run")}
    lines = ["# LIF可塑性ネットワークの実行結果", "", f"集計日時：{now()}",
             f"細胞数：{cfg.n}（E={cfg.n_exc}、I={cfg.n - cfg.n_exc}）、接続確率：{cfg.probability}。",
             f"発達：{cfg.development_ms / 1000:g}秒、STDP停止後：{cfg.frozen_ms / 1000:g}秒。",
             "", f"状態内訳：{counts}", "", "| 条件 | seed | g_max | βI | 自発発火率 Hz | 状態 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    lines += [f"| {r['case_id']} | {r['seed']} | {r['gmax']} | {r['beta_inh']} | {r['rest_rate_hz']} | {r['status']} |" for r in rows]
    lines += ["", "## 読み方", "", "固定後の全期間を連結してアバランチを解析しています。チャンク境界は分割基準にしていません。",
              "LLR、ΔCr、バースト指標は記述指標です。これらだけで臨界状態・成熟・オータプス効果を確定しません。",
              "発達の長さ、時間刻み、自発発火率、複数シード、アバランチ定義への依存性を追加確認してください。",
              "同じseedの異なるパラメータ条件は同じ配線を使います。独立標本として重複計上しません。",
              "重み固定後もSTPと自発発火は継続します。保存されたfreeze_stateは完全な再開状態ではありません。",
              "", "各条件のactivity.png、metrics.csv、frozen_metrics.jsonと生スパイクを参照してください。",
              "失敗・中断・未実行は0発火として扱いません。転送状態はupload/status.jsonに別途記録します。", ""]
    (out / "report.md").write_text("\n".join(lines))
    return counts


def plot_campaign(out):
    """複数seedの記述指標を平均し、欠測は空欄として表示する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(out)
    with (out / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    cfg = NetworkSettings.model_validate(read_json(out / "settings.json"))
    for index, rate in enumerate(cfg.rest_rates_hz):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax, key in zip(axes, ("mean_rate_hz", "delta_cr", "burstiness_index")):
            values = np.full((len(cfg.beta_inh_values), len(cfg.gmax_values)), np.nan)
            for y, beta in enumerate(cfg.beta_inh_values):
                for x, g in enumerate(cfg.gmax_values):
                    group = [float(r[key]) for r in rows if r["status"] == "complete"
                             and float(r["rest_rate_hz"]) == rate and float(r["beta_inh"]) == beta
                             and float(r["gmax"]) == g and r.get(key)]
                    if group:
                        values[y, x] = np.mean(group)
            plot = ax.imshow(values, origin="lower", aspect="auto")
            ax.set(xticks=range(len(cfg.gmax_values)), xticklabels=cfg.gmax_values,
                   yticks=range(len(cfg.beta_inh_values)), yticklabels=cfg.beta_inh_values,
                   xlabel="g_max (leak ratio)", ylabel="beta_I", title=key + " (seed mean)")
            fig.colorbar(plot, ax=ax)
        fig.suptitle(f"Rest rate {rate:g} Hz; descriptive metrics, not a criticality verdict")
        fig.tight_layout()
        fig.savefig(out / f"overview_{index:02d}.png", dpi=130)
        plt.close(fig)


def upload(out, remote, timeout_s=7200, forced=lambda: False):
    out = Path(out)
    check_remote(remote)
    if not remote:
        raise ValueError("転送先を指定してください")
    directory = out / "upload"
    directory.mkdir(exist_ok=True)
    destination = remote.rstrip("/") + "/" + out.name
    run_id = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d-%H%M%S-%f")
    state = dict(status="running", destination=destination, started=now(), transfer_id=run_id)
    save_json(directory / "status.json", state)
    body_log = directory / f"{run_id}-body.log"
    command = ["rclone", "copy", str(out), destination, "--retries", "3",
               "--exclude", "/upload/**", "--exclude", "/run.log", "--exclude", "/job.json",
               "--exclude", "/status.json", "--exclude", "/.lock", "--exclude", "/STOP*"]
    body = execute(command, body_log, timeout_s, forced=forced)
    state["body"] = body
    save_json(directory / "status.json", state)
    if body["status"] != "complete":
        state.update(status="failed", ended=now())
        save_json(directory / "status.json", state)
        return state
    # 本体の転送結果とログを最終転送する。最終転送自身のログはローカルに保持する。
    final = execute(["rclone", "copy", str(out), destination, "--retries", "3",
                     "--include", "/status.json", "--include", "/report.md", "--include", "/upload/**",
                     "--exclude", f"/upload/{run_id}-final.log", "--exclude", f"/upload/{run_id}-receipt.log"],
                    directory / f"{run_id}-final.log", timeout_s, forced=forced)
    state.update(final=final, status="complete" if final["status"] == "complete" else "failed", ended=now())
    save_json(directory / "status.json", state)
    if state["status"] == "complete":
        # コピーの結果をリモート側にも残す。失敗時にローカルで成功扱いしない。
        receipt = execute(["rclone", "copyto", str(directory / "status.json"),
                           destination + "/upload/status.json", "--retries", "3"],
                          directory / f"{run_id}-receipt.log", timeout_s, forced=forced)
        state["receipt"] = receipt
        if receipt["status"] != "complete":
            state["status"] = "failed"
        save_json(directory / "status.json", state)
    return state


def manage(out, mode="run", remote=None):
    out = Path(out).resolve()
    with (out / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("同じ実行に対する処理が既に起動しています")
        for name in ("STOP", "STOP_NOW"):
            (out / name).unlink(missing_ok=True)
        save_json(out / "job.json", dict(pid=os.getpid(), session_id=os.getsid(0),
                  process_start=process_identity(os.getpid()), started=now(), mode=mode))
        cfg = NetworkSettings.model_validate(read_json(out / "settings.json"))
        options = read_json(out / "job_options.json")
        remote = options["remote"] if remote is None else remote
        if remote != options["remote"]:
            options["remote"] = remote
            save_json(out / "job_options.json", options)
        force_stop = [False]

        def signal_stop(signum, frame):
            force_stop[0] = True
            (out / "STOP_NOW").touch()

        signal.signal(signal.SIGTERM, signal_stop)
        signal.signal(signal.SIGINT, signal_stop)
        status = dict(status="running", mode=mode, started=now())
        last_heartbeat = [0.0]

        def heartbeat():
            if time.monotonic() - last_heartbeat[0] >= 1:
                last_heartbeat[0] = time.monotonic()
                attempt = status.get("attempt")
                progress = read_json(out / attempt / "progress.json", {}) if attempt else {}
                completed, total = progress.get("completed_chunks", 0), progress.get("total_chunks", 0)
                remaining = None
                if completed:
                    current_remaining = progress["elapsed_s"] * (total - completed) / completed
                    plan = read_json(out / "plan.json")
                    done = [case_outcome(out, case["case_id"]) for case in plan]
                    known = [row["elapsed_s"] for row in done if "elapsed_s" in row]
                    pending = sum(not row for row in done) - 1
                    remaining = current_remaining + max(0, pending) * (sum(known) / len(known) if known else progress["elapsed_s"] * total / completed)
                save_json(out / "status.json", {**status, "updated": now(), "estimated_remaining_s": remaining})

        def forced():
            return force_stop[0] or (out / "STOP_NOW").exists()

        try:
            if mode == "upload":
                previous = read_json(out / "status.json", {})
                if previous.get("status") in ("prepared", "running", "paused"):
                    raise ValueError("未完了の実行です。再開または明示的な手動copyで扱ってください")
                result = upload(out, remote, options["upload_timeout_s"], forced)
                save_json(out / "status.json", {**previous, "updated": now(), "upload_status": result["status"]})
                return 0 if result["status"] == "complete" else 1
            check_sources(out)
            recover_interrupted_attempts(out)
            for case in read_json(out / "plan.json"):
                if (out / "STOP").exists() or forced():
                    break
                case_id = case["case_id"]
                old = case_outcome(out, case_id)
                if old.get("status") == "complete":
                    if not verify_attempt(out / old["attempt"]):
                        raise ValueError(f"完了成果物が欠損・改変されています: {case_id}")
                    continue
                check_sources(out)
                case_dir = out / "cases" / case_id
                case_dir.mkdir(exist_ok=True)
                attempt = case_dir / f"attempt_{len(list(case_dir.glob('attempt_*.log'))) + 1:03d}"
                status.update(current_case=case_id, attempt=str(attempt.relative_to(out)))
                heartbeat()
                command = [sys.executable, "-u", str(CLI), "_case", "--run-dir", str(out),
                           "--case-id", case_id, "--attempt", str(attempt)]
                outcome = execute(command, attempt.with_suffix(".log"), cfg.case_timeout_s, forced, heartbeat)
                if outcome["status"] == "complete" and not verify_attempt(attempt):
                    outcome.update(status="failed", error="正常終了ですが必須成果物が不完全です")
                outcome.update(attempt=str(attempt.relative_to(out)), ended=now())
                save_json(case_dir / "outcome.json", outcome)
                save_json(attempt.with_suffix(".exit.json"), outcome)
                counts = summarize(out)
                status.update(counts=counts)
                print(f"{now()} {case_id}: {outcome['status']}", flush=True)
            counts = summarize(out)
            try:
                plot_campaign(out)
            except Exception as exc:
                status["analysis_error"] = str(exc)
                save_json(out / "analysis_error.json", dict(error=str(exc)))
            paused = (out / "STOP").exists() or forced()
            status.update(status="paused" if paused else "failed" if status.get("analysis_error") or counts["complete"] != len(read_json(out / "plan.json")) else "complete",
                          ended=now(), counts=counts, current_case=None)
            save_json(out / "status.json", status)
            save_json(out / "artifact_manifest.json", inventory(out, exclude=("upload", ".lock", "run.log", "job.json", "status.json", "artifact_manifest.json")))
            if remote and not paused:
                result = upload(out, remote, options["upload_timeout_s"], forced)
                status["upload_status"] = result["status"]
                save_json(out / "status.json", status)
                if result["status"] != "complete":
                    return 1
            return 0 if status["status"] == "complete" else 1
        except Exception as exc:
            if mode == "upload":
                save_json(out / "upload_error.json", dict(error=str(exc), ended=now()))
                raise
            status.update(status="failed", error=str(exc), ended=now())
            save_json(out / "status.json", status)
            raise
