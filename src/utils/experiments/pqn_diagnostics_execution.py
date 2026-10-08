"""PQN診断の計画保存とケース実行。モデル構築はrun時だけ行う。"""

from datetime import datetime
import hashlib
import multiprocessing
from pathlib import Path
import platform
import re
import sys
from zoneinfo import ZoneInfo

from src.utils.experiments.baseline_network import BaselineSettings

ROOT = Path(__file__).resolve().parents[3]


def _now():
    return datetime.now(ZoneInfo("Asia/Tokyo"))


def create_plan(settings, resolved):
    from .pqn_diagnostics import expand_cases

    cases = expand_cases(settings)
    for case in cases:
        values = dict(resolved["baseline"])
        values.update(
            dt_ms=case["dt_ms"], warmup_ms=case["warmup_ms"],
            pulse_ms=case["pulse_ms"], post_ms=case["post_ms"],
            sample_ms=case["dt_ms"] if case["kind"] == "cell" else 1.0,
        )
        if case["kind"] == "cell":
            n_exc = len(settings.amplitudes)
            values.update(n=2 * n_exc, n_exc=n_exc, probability=0.0,
                          stimulus_fraction=1.0,
                          weights=dict(ee=0, ei=0, ie=0, ii=0))
            kwargs = dict(multiplier=0.0, amplitude_vector=settings.amplitudes * 2)
        else:
            kwargs = {key: case[key] for key in (
                "multiplier", "inhibition_factor", "stimulus_seed", "no_stimulus"
            )}
        case["settings"] = BaselineSettings.model_validate(values).model_dump()
        case["kwargs"] = kwargs

    def same_protocol(a, b, ignored):
        av, bv = dict(a["settings"]), dict(b["settings"])
        for key in ignored:
            av.pop(key, None)
            bv.pop(key, None)
        return a["kind"] == b["kind"] and av == bv and a["kwargs"] == b["kwargs"]

    pairs = []
    dts = sorted(settings.dt_values_ms, reverse=True)
    for coarse_dt, fine_dt in zip(dts, dts[1:]):
        for a in cases:
            if a["dt_ms"] != coarse_dt or a["warmup_ms"] != settings.warmup_ms:
                continue
            matches = [b for b in cases if b["dt_ms"] == fine_dt
                       and same_protocol(a, b, ("dt_ms", "sample_ms"))]
            if len(matches) != 1:
                raise ValueError(f"dt比較の相手が一意に決まりません: {a['case_id']}")
            pairs.append(dict(kind="dt", coarse=a["case_id"], fine=matches[0]["case_id"],
                              primary=fine_dt == min(dts)))
    for b in cases:
        if b["warmup_ms"] != settings.warmup_check_ms:
            continue
        matches = [a for a in cases if a["warmup_ms"] == settings.warmup_ms
                   and same_protocol(a, b, ("warmup_ms",))]
        if len(matches) != 1:
            raise ValueError(f"待機比較の相手が一意に決まりません: {b['case_id']}")
        pairs.append(dict(kind="warmup", coarse=matches[0]["case_id"],
                          fine=b["case_id"], primary=True))
    return dict(version=1, settings=settings.model_dump(), resolved=resolved,
                criteria=settings.criteria.model_dump(), cases=cases,
                comparison_pairs=pairs, created_at=_now().isoformat())


def _read_json(path, default=None):
    import json
    return json.loads(path.read_text()) if path.exists() else default


def _case_record(case_id, directory, exc=None):
    state = _read_json(directory / "status.json", {})
    retries = _read_json(directory / "build_retries.json", [])
    worker = _read_json(directory / "worker_error.json")
    error = str(exc) if exc is not None else None
    match = re.search(r"exitcode=(-?\d+)", str(state.get("error", "")))
    exitcode = int(match.group(1)) if match else (0 if exc is None else None)
    if isinstance(exc, FloatingPointError):
        status = "non_finite"
    elif match and exitcode != 0:
        status = "crashed"
    elif exc is not None or state.get("status") != "complete":
        status = "error"
    else:
        status = "complete"
    return dict(
        case_id=case_id, status=status, case_dir=str(directory),
        exception_type=type(exc).__name__ if exc else None, error=error,
        phase=state.get("phase", "unknown"), exitcode=exitcode,
        log=str(directory.with_suffix(".log")), retry_count=len(retries),
        retries=retries, status_data=state, worker_error=worker,
    )


def run_experiment(config_path, output_root, selected=None):
    from .pqn_diagnostics import (
        _atomic_json, _git_info, load_settings, analyze_run,
    )
    from .baseline_network import make_snapshot, run_case

    settings, resolved = load_settings(config_path)
    plan = create_plan(settings, resolved)
    cases = plan["cases"]
    known = [case["case_id"] for case in cases]
    if selected is not None and (not selected or set(selected) - set(known)):
        raise ValueError("空または不明なcase-id指定です")
    if selected is not None and len(set(selected)) != len(selected):
        raise ValueError("case-idの指定が重複しています")
    selected_ids = set(known if selected is None else selected)
    plan["selected_case_ids"] = [cid for cid in known if cid in selected_ids]
    run_id = _now().strftime("%Y%m%d-%H%M%S-%f")
    out = Path(output_root).resolve() / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "cases").mkdir()
    _atomic_json(out / "plan.json", plan)
    source_paths = [
        Path(__file__), ROOT / "src/utils/experiments/pqn_diagnostics.py",
        ROOT / "scripts/pqn_diagnostics.py", Path(config_path).resolve(),
        Path(resolved["baseline_path"]),
    ]
    manifest = dict(
        run_id=run_id, started_at=_now().isoformat(), status="running",
        partial=len(selected_ids) < len(known), command=list(sys.argv),
        git=_git_info(), python=platform.python_version(),
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    )
    results = [dict(case_id=cid, status="pending" if cid in selected_ids else "not_selected")
               for cid in known]

    def save_progress():
        manifest["counts"] = {name: sum(r["status"] == name for r in results) for name in (
            "complete", "non_finite", "crashed", "error", "not_selected", "pending"
        )}
        _atomic_json(out / "case_results.json", results)
        _atomic_json(out / "manifest.json", manifest)

    save_progress()
    print(f"診断出力先: {out}", flush=True)
    snapshots = {}
    original_children = {child.pid for child in multiprocessing.active_children()}
    try:
        for index, case in enumerate(cases):
            cid = case["case_id"]
            if cid not in selected_ids:
                continue
            cfg = BaselineSettings.model_validate(case["settings"])
            directory = out / "cases" / cid
            try:
                kind = case["kind"]
                if kind not in snapshots:
                    snapshot = out / f"{kind}_wiring.npz"
                    make_snapshot(cfg, settings.wiring_seed, snapshot)
                    snapshots[kind] = snapshot
                print(f"実行開始: {cid}", flush=True)
                run_case(cfg, snapshots[kind], directory,
                         ROOT / "genn_code" / "pqn_diagnostics" / run_id / cid,
                         **case["kwargs"])
                results[index] = _case_record(cid, directory)
            except Exception as exc:
                results[index] = _case_record(cid, directory, exc)
            save_progress()
            print(f"実行終了: {cid}: {results[index]['status']}", flush=True)
    except (KeyboardInterrupt, SystemExit):
        # spawnのjoin中に親だけが中断された場合も、今回の子を残さない。
        for child in multiprocessing.active_children():
            if child.pid not in original_children:
                child.terminate()
                child.join(timeout=5)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=5)
        manifest.update(status="interrupted", ended_at=_now().isoformat())
        save_progress()
        raise

    failed = any(r["status"] in ("non_finite", "crashed", "error", "pending") for r in results)
    manifest.update(status="failed" if failed else "partial" if manifest["partial"] else "complete",
                    ended_at=_now().isoformat(), analysis_status="running")
    save_progress()
    try:
        analyze_run(out)
    except Exception as exc:
        manifest.update(analysis_status="failed", analysis_error=str(exc))
        save_progress()
        raise
    manifest["analysis_status"] = "complete"
    save_progress()
    return out, int(failed)


def validate_comparison_pair(run, plan, pair):
    """設定・保存入力・配線の一致を確認し、異なる条件の比較を拒否する。"""
    import numpy as np
    import yaml

    cases = {case["case_id"]: case for case in plan["cases"]}
    left, right = cases[pair["coarse"]], cases[pair["fine"]]
    if left["kind"] != right["kind"] or left["kwargs"] != right["kwargs"]:
        raise ValueError("比較ケースの種類または実行引数が異なります")
    if pair["kind"] not in ("dt", "warmup"):
        raise ValueError("未知の比較種類です")
    values = []
    metadata = []
    directories = []
    for case in (left, right):
        directory = Path(run) / "cases" / case["case_id"]
        actual = BaselineSettings.model_validate(
            yaml.safe_load((directory / "settings.yaml").read_text())
        ).model_dump()
        if actual != case["settings"]:
            raise ValueError("実行設定が保存計画と一致しません")
        values.append(actual)
        metadata.append(_read_json(directory / "metadata.json"))
        directories.append(directory)
    ignored = ("dt_ms", "sample_ms") if pair["kind"] == "dt" else ("warmup_ms",)
    for value in values:
        for key in ignored:
            value.pop(key)
    if values[0] != values[1]:
        raise ValueError("比較軸以外の設定が異なります")
    for key in ("snapshot_fingerprint", "connectivity_hash", "absolute_weights", "source_hashes"):
        if key not in metadata[0] or metadata[0][key] != metadata[1].get(key):
            raise ValueError(f"比較軸以外の実行記録が異なります: {key}")
    for name in ("stimulus_ids.npy", "stimulus_amplitudes.npy"):
        a, b = [np.load(d / name, allow_pickle=False) for d in directories]
        if not np.array_equal(a, b):
            raise ValueError(f"比較ケースの入力が異なります: {name}")
    if _read_json(directories[0] / "neuron_constants.json") != _read_json(
        directories[1] / "neuron_constants.json"
    ):
        raise ValueError("ニューロン定数が異なります")
