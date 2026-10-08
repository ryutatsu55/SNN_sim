"""PQN応答・時間刻み診断。plan/analyze は GeNN を読み込まない。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[3]
AMPLITUDES = [0.0, -0.2, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Criteria(Strict):
    warmup_drift_abs: float = Field(default=1e-5, gt=0)
    cell_latency_abs_ms: float = Field(default=0.1, ge=0)
    cell_waveform_nrmse: float = Field(default=0.05, ge=0)
    network_count_relative: float = Field(default=0.05, ge=0)
    network_last_spike_abs_ms: float = Field(default=1.0, ge=0)


class DiagnosticsSettings(Strict):
    baseline_config: str = "baseline_network.yaml"
    backend: str = "cpu"
    dt_values_ms: list[float] = [0.05, 0.025, 0.0125, 0.00625]
    warmup_ms: float = 2000.0
    warmup_check_ms: float = 4000.0
    amplitudes: list[float] = AMPLITUDES
    pulse_widths_ms: list[float] = [20.0, 500.0]
    cell_post_ms: float = 100.0
    network_post_ms: float = 4000.0
    network_multiplier: float = 2.0
    network_inhibition_factor: float = 0.5
    wiring_seed: int = 11
    stimulus_seed: int = 101
    criteria: Criteria = Field(default_factory=Criteria)

    @model_validator(mode="after")
    def valid(self):
        if self.backend != "cpu":
            raise ValueError("診断標準設定はCPU逐次実行です")
        if len(set(self.dt_values_ms)) != len(self.dt_values_ms) or len(self.dt_values_ms) < 2:
            raise ValueError("dt_values_msは重複なしで2件以上必要です")
        if any(not np.isfinite(x) or x <= 0 for x in self.dt_values_ms):
            raise ValueError("dtは有限・正値が必要です")
        if self.warmup_check_ms <= self.warmup_ms:
            raise ValueError("warmup_check_msはwarmup_msより長くしてください")
        if not self.amplitudes or len(set(self.amplitudes)) != len(self.amplitudes):
            raise ValueError("振幅は空でなく重複なしが必要です")
        if 0 not in self.amplitudes:
            raise ValueError("振幅0の対照が必要です")
        if 500.0 not in self.pulse_widths_ms or len(set(self.pulse_widths_ms)) != len(self.pulse_widths_ms):
            raise ValueError("パルス幅は500 msを含む重複なしの列が必要です")
        dts = sorted(self.dt_values_ms, reverse=True)
        if any(not np.isclose(a / b, round(a / b), rtol=0, atol=1e-8)
               for a, b in zip(dts, dts[1:])):
            raise ValueError("共通時刻で比較するため隣接dtの比は整数が必要です")
        return self


def load_settings(path: str | Path) -> tuple[DiagnosticsSettings, dict]:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    settings = DiagnosticsSettings.model_validate(raw)
    base_path = (path.parent / settings.baseline_config).resolve()
    from src.utils.experiments.baseline_network import BaselineSettings
    base = BaselineSettings.model_validate(yaml.safe_load(base_path.read_text()))
    values = base.model_dump()
    values.update(backend="cpu", dt_ms=min(settings.dt_values_ms), warmup_ms=settings.warmup_ms,
                  pulse_ms=20.0, post_ms=settings.network_post_ms, wiring_seeds=[settings.wiring_seed],
                  stimulus_seeds=[settings.stimulus_seed], multipliers=[0.0, settings.network_multiplier],
                  inhibition_factors=[settings.network_inhibition_factor], sample_ms=1.0)
    # BaselineSettingsによる最終検証。個々のcaseでもdt/期間を検証する。
    resolved = BaselineSettings.model_validate(values)
    return settings, {"baseline_path": str(base_path), "baseline": resolved.model_dump()}


def _tag(x: float) -> str:
    return (f"{x:g}").replace("-", "m").replace(".", "p")


def expand_cases(s: DiagnosticsSettings) -> list[dict]:
    cases = []
    for pulse in s.pulse_widths_ms:
        for dt in s.dt_values_ms:
            cases.append(dict(case_id=f"cell_p{_tag(pulse)}_dt{_tag(dt)}_w{_tag(s.warmup_ms)}",
                              kind="cell", dt_ms=dt, warmup_ms=s.warmup_ms, pulse_ms=pulse,
                              post_ms=s.cell_post_ms))
    dt = min(s.dt_values_ms)
    cases.append(dict(case_id=f"cell_p500_dt{_tag(dt)}_w{_tag(s.warmup_check_ms)}", kind="cell",
                      dt_ms=dt, warmup_ms=s.warmup_check_ms, pulse_ms=500.0, post_ms=s.cell_post_ms))
    for dt in s.dt_values_ms:
        for label, mult, no_stim in (("no_stim", s.network_multiplier, True),
                                     ("zero_weight", 0.0, False), ("stim", s.network_multiplier, False)):
            cases.append(dict(case_id=f"net_{label}_dt{_tag(dt)}_w{_tag(s.warmup_ms)}", kind="network",
                              dt_ms=dt, warmup_ms=s.warmup_ms, pulse_ms=20.0,
                              post_ms=s.network_post_ms, multiplier=mult,
                              inhibition_factor=s.network_inhibition_factor,
                              no_stimulus=no_stim, stimulus_seed=s.stimulus_seed))
    dt = min(s.dt_values_ms)
    cases.append(dict(case_id=f"net_stim_dt{_tag(dt)}_w{_tag(s.warmup_check_ms)}", kind="network",
                      dt_ms=dt, warmup_ms=s.warmup_check_ms, pulse_ms=20.0,
                      post_ms=s.network_post_ms, multiplier=s.network_multiplier,
                      inhibition_factor=s.network_inhibition_factor, no_stimulus=False,
                      stimulus_seed=s.stimulus_seed))
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("ケースIDが重複しています")
    return cases


def threshold_crossings(v: np.ndarray, initial_v: float, threshold: float) -> np.ndarray:
    old = np.r_[initial_v, np.asarray(v)[:-1]]
    return np.flatnonzero((np.asarray(v) >= threshold) & (old < threshold))


def cell_rows(case: dict, directory: Path, amplitudes=AMPLITUDES) -> list[dict]:
    with np.load(directory / "traces.npz") as tr, np.load(directory / "spikes.npz") as sp:
        dt = case["dt_ms"]
        times, ids = sp["times_ms"], sp["ids"]
        states = {key: tr[key] for key in ("V", "N", "Q")}
        end_times = tr["time_end_ms"]
        expected_steps = round((case["warmup_ms"] + case["pulse_ms"] + case["post_ms"]) / dt)
        if len(end_times) != expected_steps or not np.allclose(
            end_times / dt, np.arange(1, expected_steps + 1), atol=1e-6, rtol=0
        ):
            raise ValueError("独立細胞の全ステップ波形がありません")
        if not np.array_equal(tr["ids"], np.arange(2 * len(amplitudes))):
            raise ValueError("独立細胞の記録IDが一致しません")
        if any(v.shape != (expected_steps, 2 * len(amplitudes)) or
               not np.all(np.isfinite(v)) for v in states.values()):
            raise ValueError("状態波形の形状または有限性が不正です")
        if not np.all(np.isfinite(times)) or not np.allclose(
            times / dt, np.rint(times / dt), atol=1e-6, rtol=0
        ):
            raise ValueError("発火時刻がステップに対応しません")
        spike_steps = np.rint(times / dt).astype(np.int64)
        # 保存波形は更新終了時刻。区間判定には整数の更新開始ステップを使う。
        steps = np.arange(expected_steps)
        warmup = round(case["warmup_ms"] / dt)
        stop = warmup + round(case["pulse_ms"] / dt)
        intervals = {"pre": (0, warmup), "during": (warmup, stop), "post": (stop, expected_steps)}
        constants = json.loads((directory / "neuron_constants.json").read_text())
        rows = []
        for neuron_id in range(2 * len(amplitudes)):
            mode = "RSexci" if neuron_id < len(amplitudes) else "RSinhi"
            group = "Exc" if mode == "RSexci" else "Inh"
            amp = amplitudes[neuron_id % len(amplitudes)]
            own = np.sort(spike_steps[ids == neuron_id])
            if np.any((own < 0) | (own >= expected_steps)):
                raise ValueError("観測区間外の発火です")
            threshold = float(constants[group]["params"]["V_THRESH"])
            v = states["V"][:, neuron_id]
            crossings = threshold_crossings(v, float(constants[group]["initial_vars"]["V"]), threshold)
            during = own[(own >= warmup) & (own < stop)]
            row = dict(
                case_id=case["case_id"], mode=mode, amplitude=amp, dt_ms=dt,
                warmup_ms=case["warmup_ms"], pulse_ms=case["pulse_ms"],
                first_latency_ms=float((during[0] - warmup) * dt) if during.size else None,
                threshold_fraction=float(np.mean(v[warmup:] >= threshold)),
                recorded_crossings=int(crossings.size), spike_events=int(own.size),
                crossing_match=bool(np.array_equal(crossings, own)),
                zero_input_spikes=int(own.size) if amp == 0 else None,
            )
            for phase, (lo, hi) in intervals.items():
                row[f"{phase}_spikes"] = int(np.count_nonzero((own >= lo) & (own < hi)))
                row[f"{phase}_threshold_fraction"] = float(np.mean(v[lo:hi] >= threshold))
                for var, array in states.items():
                    values = array[lo:hi, neuron_id]
                    for suffix, fn in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)):
                        row[f"{phase}_{var}_{suffix}"] = float(fn(values))
            tail_start = max(warmup, stop - round(100 / dt))
            for var, array in states.items():
                tail = array[tail_start:stop, neuron_id]
                row[f"input_tail_{var}_mean"] = float(np.mean(tail))
                row[f"input_tail_{var}_std"] = float(np.std(tail))
            rows.append(row)
        return rows


def _atomic_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    os.replace(tmp,path)


def _write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _waveform_error(run: Path, coarse: dict, fine: dict) -> dict:
    directories = [run / "cases" / c["case_id"] for c in (coarse, fine)]
    with np.load(directories[0] / "traces.npz") as a, np.load(directories[1] / "traces.npz") as b:
        ta = a["time_end_ms"] - coarse["warmup_ms"]
        tb = b["time_end_ms"] - fine["warmup_ms"]
        # 最初の刺激更新後から観測終了まで。待機区間で誤差を薄めない。
        coarse_steps = np.rint(ta / fine["dt_ms"]).astype(np.int64)
        fine_steps = np.rint(tb / fine["dt_ms"]).astype(np.int64)
        limit = round((coarse["pulse_ms"] + coarse["post_ms"]) / fine["dt_ms"])
        ia = np.flatnonzero((coarse_steps > 0) & (coarse_steps <= limit))
        ib = np.searchsorted(fine_steps, coarse_steps[ia])
        if (not ia.size or np.any(ib >= len(fine_steps)) or
                not np.array_equal(fine_steps[ib], coarse_steps[ia]) or
                not np.allclose(ta[ia], tb[ib], atol=1e-8, rtol=0) or
                not np.array_equal(a["ids"], b["ids"])):
            raise ValueError("比較波形の共通時刻・細胞IDが一致しません")
        errors = {}
        for name in ("V", "N", "Q"):
            x, y = a[name][ia], b[name][ib]
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                raise ValueError("比較波形に非有限値があります")
            errors[name] = np.sqrt(np.mean((x - y) ** 2, axis=0)) / np.maximum(1, np.max(np.abs(y), axis=0))
        return errors


def _cell_comparison(pair: dict, plan: dict, run: Path, by_case: dict) -> list[dict]:
    left,right=by_case.get(pair["coarse"]),by_case.get(pair["fine"])
    if left is None or right is None:
        return [dict(**pair,category="cell",status="inconclusive",reason="比較対象ケースが欠測")]
    cases={c["case_id"]:c for c in plan["cases"]}; a_case,b_case=cases[pair["coarse"]],cases[pair["fine"]]
    a_map={(r["mode"],r["amplitude"]):r for r in left}; b_map={(r["mode"],r["amplitude"]):r for r in right}
    criteria=plan["criteria"]; result=[]
    waveform_errors=_waveform_error(run,a_case,b_case)
    for key in sorted(set(a_map)&set(b_map)):
        a,b=a_map[key],b_map[key]; diffs={}; passed=True
        for phase in ("pre","during","post"):
            diffs[f"{phase}_spikes_delta"]=a[f"{phase}_spikes"]-b[f"{phase}_spikes"]
            passed &= a[f"{phase}_spikes"]==b[f"{phase}_spikes"]
        if a["first_latency_ms"] is None or b["first_latency_ms"] is None:
            latency=None if a["first_latency_ms"] is None and b["first_latency_ms"] is None else float("inf")
        else: latency=a["first_latency_ms"]-b["first_latency_ms"]
        diffs["first_latency_delta_ms"]=None if latency is None else (latency if np.isfinite(latency) else None)
        if latency is not None: passed &= np.isfinite(latency) and abs(latency)<=criteria["cell_latency_abs_ms"]
        passed &= a["crossing_match"] and b["crossing_match"]
        if not a["crossing_match"] or not b["crossing_match"]: diffs["crossing_recording_consistent"]=False
        cell_index=plan["settings"]["amplitudes"].index(key[1])+(len(plan["settings"]["amplitudes"]) if key[0]=="RSinhi" else 0)
        cell_errors={k:(None if v is None else float(v[cell_index])) for k,v in waveform_errors.items()}
        diffs.update({f"{k}_nrmse":v for k,v in cell_errors.items()})
        passed &= all(v is not None and v<=criteria["cell_waveform_nrmse"] for v in cell_errors.values())
        result.append(dict(**pair,category="cell",mode=key[0],amplitude=key[1],status="match" if passed else "mismatch",differences=diffs))
    return result or [dict(**pair,category="cell",status="inconclusive",reason="対応指標なし")]


def _network_comparison(pair: dict, plan: dict, rows: dict) -> dict:
    a,b=rows.get(pair["coarse"]),rows.get(pair["fine"])
    if a is None or b is None: return dict(**pair,category="network",status="inconclusive",reason="比較対象ケースが欠測")
    criteria=plan["criteria"]; passed=True; diffs={}
    passed &= a["activity_status"]==b["activity_status"]
    diffs["activity_status_equal"]=a["activity_status"]==b["activity_status"]
    for key in ("post_spikes","post_exc_spikes","post_inh_spikes","post_participants","tail_spikes"):
        x,y=a[key],b[key]; relative=abs(x-y)/max(1,abs(y)); diffs[key]={"delta":x-y,"relative":relative}
        passed &= relative<=criteria["network_count_relative"]
    x,y=a["last_post_latency_ms"],b["last_post_latency_ms"]
    delta=None if x is None and y is None else (None if x is None or y is None else x-y)
    diffs["last_post_latency_delta_ms"]=delta
    if x is None or y is None: passed &= x is None and y is None
    else: passed &= abs(delta)<=criteria["network_last_spike_abs_ms"]
    if a["no_stimulus"]:
        zero_a=all(a[k]==0 for k in ("pre_spikes","during_spikes","post_spikes"))
        zero_b=all(b[k]==0 for k in ("pre_spikes","during_spikes","post_spikes"))
        diffs["no_stimulus_zero_spikes"]={"coarse":zero_a,"fine":zero_b}; passed &= zero_a and zero_b
    if a["multiplier"]==0:
        diffs["zero_weight_during_spikes_delta"]=a["during_spikes"]-b["during_spikes"]
        passed &= a["during_spikes"]==b["during_spikes"]
    if pair["kind"]=="warmup":
        drift_ok=True
        for row in (a,b):
            drift=row.get("warmup_drift")
            drift_ok &= isinstance(drift,dict) and bool(drift) and all(v is not None and np.isfinite(v) and v<=criteria["warmup_drift_abs"] for v in drift.values())
        diffs["warmup_drift_below_criterion"]=bool(drift_ok); passed &= drift_ok
    return dict(**pair,category="network",status="match" if passed else "mismatch",differences=diffs)


def _git_info():
    def run(args):
        return subprocess.run(args,cwd=ROOT,text=True,capture_output=True).stdout.strip()
    return {"head":run(["git","rev-parse","HEAD"]),"status":run(["git","status","--short"])}


def _source_hashes():
    files=[ROOT/"src/utils/experiments/pqn_diagnostics.py",ROOT/"src/utils/experiments/pqn_diagnostics_execution.py",ROOT/"scripts/pqn_diagnostics.py"]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


from src.utils.experiments.pqn_diagnostics_execution import create_plan, run_experiment


def _plot_results(plan, run, completed, network_rows):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amps = plan["settings"].get("amplitudes", [])
    if amps:
        index = amps.index(0.4) if 0.4 in amps else 0
        for pulse in plan["settings"]["pulse_widths_ms"]:
            selected = [c for c in plan["cases"] if c["kind"] == "cell" and
                        c["pulse_ms"] == pulse and c["warmup_ms"] == plan["settings"]["warmup_ms"]
                        and c["case_id"] in completed]
            if not selected:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
            for case in selected:
                with np.load(run / "cases" / case["case_id"] / "traces.npz") as traces:
                    t = traces["time_end_ms"] - case["warmup_ms"]
                    mask = (t >= -5) & (t <= pulse + case["post_ms"])
                    for column, var in enumerate(("V", "N", "Q")):
                        values = traces[var]
                        for group in range(2):
                            axes[group, column].plot(t[mask], values[mask, index + group * len(amps)],
                                                     label=f"dt={case['dt_ms']:g}", lw=0.7)
            for group in range(2):
                for column, var in enumerate(("V", "N", "Q")):
                    ax = axes[group, column]
                    ax.axvspan(0, pulse, color="orange", alpha=0.12)
                    ax.set_title(f"{('RSexci', 'RSinhi')[group]} {var}; input={amps[index]:g}")
                    ax.set_xlabel("Time from pulse onset (ms)")
                    ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(run / f"cell_waveforms_p{_tag(pulse)}.png", dpi=150)
            plt.close(fig)
    if network_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for label in ("stim", "no_stim", "zero_weight"):
            rows = [r for r in network_rows if r["warmup_ms"] == plan["settings"]["warmup_ms"]
                    and ("no_stim" if r["no_stimulus"] else "zero_weight" if r["multiplier"] == 0 else "stim") == label]
            rows = sorted(rows, key=lambda r: r["dt_ms"])
            for ax, key in zip(axes, ("post_spikes", "tail_spikes")):
                ax.scatter([r["dt_ms"] for r in rows], [r[key] for r in rows], label=label)
                ax.set(xlabel="dt (ms)", ylabel=key)
                ax.legend()
        fig.tight_layout()
        fig.savefig(run / "network_counts.png", dpi=150)
        plt.close(fig)


def _quality_check(directory, case, cell_metrics, network_metrics, criteria):
    drift = json.loads((directory / "warmup_drift.json").read_text())
    if not isinstance(drift, dict) or not all(
        key in drift and drift[key] is not None and np.isfinite(drift[key]) for key in ("V", "N", "Q")
    ):
        return dict(status="inconclusive", reason="待機状態の記録が欠測", drift=drift)
    checks = {"warmup_stationary": all(0 <= drift[k] <= criteria["warmup_drift_abs"] for k in ("V", "N", "Q"))}
    if case["kind"] == "cell":
        checks["crossings_match"] = all(r["crossing_match"] for r in cell_metrics)
        checks["no_pre_spikes"] = all(r["pre_spikes"] == 0 for r in cell_metrics)
        checks["zero_input_silent"] = all(r["zero_input_spikes"] == 0 for r in cell_metrics if r["amplitude"] == 0)
    else:
        checks["no_pre_spikes"] = network_metrics["pre_spikes"] == 0
        if case["no_stimulus"]:
            checks["zero_input_silent"] = all(network_metrics[k] == 0 for k in ("pre_spikes", "during_spikes", "post_spikes"))
            checks["zero_input_current"] = bool(np.all(np.load(directory / "stimulus_amplitudes.npy") == 0))
    return dict(status="match" if all(checks.values()) else "mismatch", checks=checks, drift=drift)


def _aggregate(comparisons):
    if not comparisons or any(c["status"] == "inconclusive" for c in comparisons):
        return "inconclusive"
    return "mismatch" if any(c["status"] == "mismatch" for c in comparisons) else "match"


def analyze_run(run_dir: str | Path) -> dict:
    from .pqn_diagnostics_execution import validate_comparison_pair

    run = Path(run_dir)
    plan = json.loads((run / "plan.json").read_text())
    results = json.loads((run / "case_results.json").read_text())
    cases = {c["case_id"]: c for c in plan["cases"]}
    cell_by_case, network_by_case, quality, analysis_errors = {}, {}, {}, {}
    cell_rows_all, network_rows_all = [], []
    completed = set()
    manual_stops = []
    for result in results:
        cid = result["case_id"]
        intervention = run / "cases" / cid / "manual_stop.json"
        if intervention.exists():
            manual_stops.append(json.loads(intervention.read_text()))
            # 元のrun_case記録は保管。派生集計では人為的停止をクラッシュから分離する。
            result = dict(result, status="error")
            for item in results:
                if item["case_id"] == cid:
                    item["status"] = "error"
        if result["status"] != "complete":
            continue
        case = cases[cid]
        directory = run / "cases" / cid
        try:
            if json.loads((directory / "status.json").read_text())["status"] != "complete":
                raise ValueError("case_resultsと単条件statusが不一致です")
            if case["kind"] == "cell":
                rows = cell_rows(case, directory, plan["settings"]["amplitudes"])
                metrics = None
            else:
                rows = []
                metrics = json.loads((directory / "metrics.json").read_text())
                metrics = dict(metrics, case_id=cid, dt_ms=case["dt_ms"], warmup_ms=case["warmup_ms"],
                               no_stimulus=case["no_stimulus"], multiplier=case["multiplier"],
                               warmup_drift=json.loads((directory / "warmup_drift.json").read_text()))
            quality[cid] = _quality_check(directory, case, rows, metrics, plan["criteria"])
            if rows:
                cell_by_case[cid] = rows
                cell_rows_all.extend(rows)
            else:
                network_by_case[cid] = metrics
                network_rows_all.append(metrics)
            completed.add(cid)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            analysis_errors[cid] = str(exc)
    _write_csv(run / "cell_metrics.csv", cell_rows_all)
    _write_csv(run / "network_metrics.csv", network_rows_all)
    comparisons = []
    for pair in plan.get("comparison_pairs", []):
        category = cases[pair["coarse"]]["kind"]
        try:
            if pair["coarse"] in completed and pair["fine"] in completed:
                validate_comparison_pair(run, plan, pair)
            if category == "cell":
                compared = _cell_comparison(pair, plan, run, cell_by_case)
            else:
                compared = [_network_comparison(pair, plan, network_by_case)]
            for comparison in compared:
                checks = {cid: quality[cid] for cid in (pair["coarse"], pair["fine"]) if cid in quality}
                comparison["quality_checks"] = checks
                if any(c["status"] == "inconclusive" for c in checks.values()):
                    comparison["status"] = "inconclusive"
                elif comparison["status"] != "inconclusive" and any(c["status"] == "mismatch" for c in checks.values()):
                    comparison["status"] = "mismatch"
            comparisons.extend(compared)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            comparisons.append(dict(pair, category=category, status="inconclusive", reason=str(exc)))
    _atomic_json(run / "comparisons.json", comparisons)
    _write_csv(run / "comparisons.csv", comparisons)
    _plot_results(plan, run, completed, network_rows_all)
    counts = {key: sum(r["status"] == key for r in results) for key in (
        "complete", "non_finite", "crashed", "error", "not_selected", "pending"
    )}
    primary = [c for c in comparisons if c.get("primary")]
    summary = {status: sum(c["status"] == status for c in comparisons) for status in ("match", "mismatch", "inconclusive")}
    decision = dict(
        counts=counts, partial=counts["not_selected"] > 0,
        status=_aggregate(primary), primary_dt_status=_aggregate([c for c in primary if c["kind"] == "dt"]),
        warmup_status=_aggregate([c for c in primary if c["kind"] == "warmup"]),
        comparison_counts=summary, comparisons=comparisons, quality_checks=quality,
        analysis_errors=analysis_errors, manual_stops=manual_stops,
    )
    if analysis_errors or any(counts[k] for k in ("non_finite", "crashed", "error", "pending", "not_selected")):
        decision["status"] = "inconclusive"
    _atomic_json(run / "decision.json", decision)
    labels = {"match": "暫定基準で一致", "mismatch": "暫定基準で不一致", "inconclusive": "判定不能"}
    lines = ["# PQN応答・時間刻み診断", "",
             f"ケース：正常終了{counts['complete']}、非有限{counts['non_finite']}、クラッシュ{counts['crashed']}、その他失敗{counts['error']}（手動停止{len(manual_stops)}を含む）、未選択{counts['not_selected']}、未実行{counts['pending']}。", "",
             f"最細2刻みの主比較：{labels[decision['primary_dt_status']]}。待機延長比較：{labels[decision['warmup_status']]}。", "",
             "基準は実行前に決めた探索用の暫定値です。収束の証明・モデルの生物学的妥当性・臨界性の確定を意味しません。", "",
             "| 比較 | 条件 | 判定 |", "| --- | --- | --- |"]
    for item in primary:
        detail = f"{item.get('mode', 'network')} {item.get('amplitude', '')}"
        lines.append(f"| {item['coarse']} → {item['fine']} | {detail} | {labels[item['status']]} |")
    lines.extend(["", "## RSinhiの長時間入力", ""])
    for row in cell_rows_all:
        if row["mode"] == "RSinhi" and row["pulse_ms"] == 500 and row["amplitude"] == 0.4:
            lines.append(f"- {row['case_id']}：入力中{row['during_spikes']}発、入力末尾V平均{row['input_tail_V_mean']:.6g}、V標準偏差{row['input_tail_V_std']:.6g}、閾値交差との一致={row['crossing_match']}。")
    lines.extend(["", "## 次の判断", ""])
    if decision["status"] == "inconclusive":
        lines.append("欠測・実行失敗が残るため、一式の数値判定は未完了。失敗コードとログを確認し、再実行は別実行として記録する。成功した比較の差はcomparisons.csvを参照する。")
    elif decision["status"] == "mismatch":
        lines.append("時間刻み・待機時間または静止・記録の確認基準を満たしていない。該当する比較を絞って検討する。")
    else:
        lines.append("今回の暫定数値基準を満たした。RSinhiの原資料・モード選択との対応を確認した後に重み探索を計画する。")
    if analysis_errors:
        lines.extend(["", "## 読み取り・比較の問題", "", *[f"- {k}: {v}" for k, v in analysis_errors.items()]])
    (run / "report.md").write_text("\n".join(lines) + "\n")
    _atomic_json(run / "analysis_manifest.json", dict(
        analyzed_at=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(), source_hashes=_source_hashes(),
        input_hashes={name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in ("plan.json", "case_results.json")},
    ))
    return decision
