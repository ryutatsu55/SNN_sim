"""LIF・PQNの独立単一細胞実験：校正、二連刺激、オータプス走査。"""

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pygenn
import yaml

from src.core.config_manager import NeuronConfig
from src.core.registry import NEURON_MODELS
from src.utils.autapse_analysis import (
    matched_dendritic_delay,
    spike_metrics,
    steps_for,
    threshold_bracket,
)
import src.models.neurons.lif  # noqa: F401
import src.models.neurons.pqn_float  # noqa: F401


@dataclass(frozen=True)
class Case:
    label: str
    shape: str
    amplitude: float
    width_ms: float
    weight: float = 0.0
    delay_ms: float = 0.0
    weight_factor: float = 0.0
    second_gap_ms: float = -1.0


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def save_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_config(cfg):
    # 基本時間と刺激条件は有限の正値にする。
    for name in (
        "dt_ms",
        "warmup_ms",
        "post_ms",
        "short_ms",
        "long_ms",
        "tau_syn_ms",
        "tail_ms",
        "burst_isi_ms",
        "stimulus_factor",
    ):
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} は有限の正値が必要です")

    if cfg["tail_ms"] > cfg["post_ms"] or cfg["stimulus_factor"] <= 1:
        raise ValueError("末尾窓は観測時間以下、刺激係数は1より大きくしてください")

    # 探索点数と発火数の閾値を確認する。
    for name in ("threshold_points", "burst_min_spikes"):
        if not isinstance(cfg[name], int) or cfg[name] < 2:
            raise ValueError(f"{name} は2以上の整数が必要です")

    # 各時間がdtの整数倍で表せることを確認する。
    for name in ("warmup_ms", "post_ms", "short_ms", "long_ms", "tail_ms"):
        steps_for(cfg[name], cfg["dt_ms"], name)
    steps_for(20 * cfg["tau_syn_ms"], cfg["dt_ms"], "exponential cutoff")

    # 重み、遅延、回復間隔の探索値を検証する。
    for name in ("weight_factors", "delays_ms", "recovery_gaps_ms"):
        values = np.asarray(cfg[name], dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{name} は有限の非負リストが必要です")
        if np.unique(values).size != values.size:
            raise ValueError(f"{name} に重複があります")
        if name != "weight_factors":
            for value in values:
                steps_for(value, cfg["dt_ms"], name)

    if 0 not in cfg["weight_factors"] or min(cfg["recovery_gaps_ms"]) <= 0:
        raise ValueError("重み0の対照と、正の二連入力間隔が必要です")

    # 比較対象モデルと振幅探索範囲を検証する。
    for spec in cfg["models"].values():
        if (spec["type"], spec["mode"]) not in (
            ("LIF", "excitatory"),
            ("PQN_float", "RSexci"),
        ):
            raise ValueError("今回の比較対象はLIF/excitatoryとPQN_float/RSexciです")

        lo, hi = spec["amplitude_min"], spec["amplitude_max"]
        psc_hi = spec.get("psc_amplitude_max", hi)
        if not all(np.isfinite([lo, hi, psc_hi])) or not (0 < lo < hi and lo < psc_hi):
            raise ValueError("振幅探索範囲が不正です")


def delayed_pulse_model():
    """組み込み遅延シナプスと同式で32bitの遅延ステップを使う。"""
    return pygenn.create_weight_update_model(
        "ComparisonDelayedPulse",
        vars=[
            ("g", "scalar", pygenn.VarAccess.READ_ONLY),
            ("d", "unsigned int", pygenn.VarAccess.READ_ONLY),
        ],
        pre_spike_syn_code="addToPostDelay(g, d);",
    )


def run_bank(name, spec, cfg, cases, out, build_dir):
    """各IDは独立した一細胞実験。自己結合以外の接続は生成しない。"""
    out.mkdir(parents=True, exist_ok=False)
    dt, start, tau = cfg["dt_ms"], cfg["warmup_ms"], cfg["tau_syn_ms"]

    profiles = yaml.safe_load((ROOT / "configs/components/neurons.yaml").read_text())
    neuron_cfg = NeuronConfig(
        type=spec["type"],
        mode=spec["mode"],
        polarity="excitatory",
        num=len(cases),
        **profiles[spec["type"]][spec["mode"]],
    )
    if spec["type"] == "LIF":
        neuron_cfg.Ioffset = 0.0
    implementation = NEURON_MODELS.get(spec["type"])(neuron_cfg, dt)

    model = pygenn.GeNNModel(
        "float", name, backend="single_threaded_cpu", time_precision="double"
    )
    model.dt = dt
    model.seed = 1
    neuron_model = implementation.model_class
    pop = model.add_neuron_population(
        "Cells", len(cases), neuron_model, implementation.params, implementation.vars
    )
    pop.spike_recording_enabled = True
    # 平均電流補正は組み込みExpCurrと同一。tは更新開始時刻。
    source = pygenn.create_current_source_model(
        "ComparisonStimulus",
        params=["Start", "Tau", "Scale"],
        vars=[
            ("Amp", "scalar"),
            ("Width", "scalar"),
            ("Gap", "scalar"),
            ("Shape", "unsigned int"),
            ("Injected", "scalar"),
        ],
        injection_code="""
        const double elapsed = t - Start;
        const double centered = elapsed + dt * 0.5;
        Injected = 0.0;
        if(centered >= 0.0 && centered < Width) {
            Injected = (Shape == 0) ? Amp : Amp * Scale * exp(-elapsed / Tau);
        }
        if(Shape == 1 && Gap >= 0.0 && centered >= Gap && centered < Gap + Width) {
            Injected += Amp * Scale * exp(-(elapsed - Gap) / Tau);
        }
        injectCurrent(Injected);
        """,
    )
    current = model.add_current_source(
        "Stimulus",
        source,
        pop,
        {"Start": start, "Tau": tau, "Scale": -tau * np.expm1(-dt / tau) / dt},
        {
            "Amp": np.array([c.amplitude for c in cases]),
            "Width": np.array([c.width_ms for c in cases]),
            "Gap": np.array([c.second_gap_ms for c in cases]),
            "Shape": np.array([int(c.shape == "exp") for c in cases], dtype=np.uint32),
            "Injected": 0.0,
        },
    )

    # 非零重みがある場合は、各条件に自己結合を1本ずつ作る。
    delay_steps = np.array(
        [steps_for(c.delay_ms, dt, "delay") for c in cases], dtype=np.uint32
    )
    weights = np.array([c.weight for c in cases], dtype=np.float32)
    if np.any(weights):
        # 組み込みStaticPulseDendriticDelayと同式。細かいdtでも折り返さない32bit遅延。
        delayed = delayed_pulse_model()
        syn = model.add_synapse_population(
            "Autapses",
            "SPARSE",
            pop,
            pop,
            pygenn.init_weight_update(delayed, {}, {"g": weights, "d": delay_steps}),
            pygenn.init_postsynaptic("ExpCurr", {"tau": tau}, {}),
        )
        ids = np.arange(len(cases), dtype=np.uint32)
        syn.set_sparse_connections(ids, ids)
        syn.max_dendritic_delay_timesteps = int(delay_steps.max()) + 1

    # 観測区間を決め、実験条件をメタデータに保存する。
    stops = np.array([start + c.width_ms + max(0.0, c.second_gap_ms) for c in cases])
    end = float(stops.max() + cfg["post_ms"])
    nsteps = steps_for(end, dt)
    metadata = {
        "model": spec,
        "neuron_config": neuron_cfg.model_dump(),
        "neuron_params": implementation.params,
        "initial_vars": implementation.vars,
        "settings": cfg,
        "cases": [asdict(c) for c in cases],
        "delay_steps": delay_steps.tolist(),
        "effective_arrival_delay_ms": ((delay_steps.astype(float) + 1) * dt).tolist(),
        "delay_note": "CPUテストで到着遅延=(d+1)*dtを確認。指定遅延に伝達更新の1dtが加わる。",
        "time_convention": "スパイク・入力は更新開始時刻、状態は更新終了時刻。区間は左閉右開。",
        "voltage_unit": "mV" if spec["type"] == "LIF" else "PQN model units",
        "input_unit": "nA" if spec["type"] == "LIF" else "PQN model units",
        "connectivity": "独立細胞。非零重みを含む実験群のみi->iの自己結合を生成。",
    }
    save_json(out / "metadata.json", metadata)

    # モデルをビルドし、状態・入力・スパイクを記録する。
    build_dir.mkdir(parents=True, exist_ok=True)
    model.build(path_to_model=str(build_dir.resolve()))
    model.load(num_recording_timesteps=nsteps)
    state_names = [key for key in ("V", "N", "Q", "U", "RefracTime") if key in pop.vars]
    traces = {
        key: np.empty((nsteps, len(cases)), dtype=np.float32) for key in state_names
    }
    inputs = np.empty((nsteps, len(cases)), dtype=np.float32)
    try:
        for step in range(nsteps):
            model.step_time()
            for key in state_names:
                pop.vars[key].pull_from_device()
                traces[key][step] = pop.vars[key].view.reshape(-1)
            current.vars["Injected"].pull_from_device()
            inputs[step] = current.vars["Injected"].view.reshape(-1)
        model.pull_recording_buffers_from_device()
        times, ids = pop.spike_recording_data[0]
        times = np.rint(np.asarray(times) / dt) * dt
        ids = np.asarray(ids).copy()
    finally:
        model.unload()

    # 記録した時系列とスパイク時刻を保存する。
    start_times = np.arange(nsteps) * dt
    np.savez_compressed(
        out / "traces.npz",
        time_start_ms=start_times,
        time_end_ms=start_times + dt,
        injected=inputs,
        **traces,
    )
    np.savetxt(
        out / "spikes.csv",
        np.column_stack((times, ids)),
        delimiter=",",
        header="time_ms,case_id",
        comments="",
    )
    if any(not np.all(np.isfinite(v)) for v in traces.values()):
        raise RuntimeError(f"非有限の状態を検出しました: {out}")

    # 条件ごとの観測終了時刻を使って指標を計算する。
    rows = []
    for i, case in enumerate(cases):
        own_end = float(stops[i] + cfg["post_ms"])
        spikes = times[(ids == i) & (times < own_end)]
        metrics = spike_metrics(
            spikes,
            start,
            float(stops[i]),
            own_end,
            cfg["tail_ms"],
            cfg["burst_isi_ms"],
            cfg["burst_min_spikes"],
        )
        second_start = start + max(case.second_gap_ms, 0)
        rows.append(
            {
                "case_id": i,
                **asdict(case),
                "stimulus_end_ms": float(stops[i]),
                "observation_end_ms": own_end,
                "spikes_after_second_onset": int(
                    np.count_nonzero(spikes >= second_start)
                ),
                **metrics,
            }
        )
    save_table(out / "summary.csv", rows)

    # 代表条件の膜電位と入力を描画する。
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    selected = np.unique(np.linspace(0, len(cases) - 1, min(4, len(cases))).astype(int))
    for i in selected:
        axes[0].plot(start_times + dt, traces["V"][:, i], label=cases[i].label, lw=0.8)
        axes[1].plot(start_times, inputs[:, i], lw=0.8)
    axes[0].set_ylabel(metadata["voltage_unit"])
    axes[1].set(xlabel="Time (ms)", ylabel=metadata["input_unit"])
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=150)
    plt.close(fig)
    print(f"完了: {name} ({len(cases)}条件)", flush=True)
    return rows


def calibrate(label, spec, cfg, out, build):
    amplitudes = np.r_[
        0.0,
        np.geomspace(
            spec["amplitude_min"], spec["amplitude_max"], cfg["threshold_points"]
        ),
    ]
    grids = {kind: amplitudes for kind in ("short", "long")}
    grids["psc"] = np.r_[
        0.0,
        np.geomspace(
            spec["amplitude_min"],
            spec.get("psc_amplitude_max", spec["amplitude_max"]),
            cfg["threshold_points"],
        ),
    ]
    cases = []
    for shape, width, kind in (
        ("rect", cfg["short_ms"], "short"),
        ("rect", cfg["long_ms"], "long"),
        ("exp", 20 * cfg["tau_syn_ms"], "psc"),
    ):
        cases.extend(
            Case(f"{kind}_{i}", shape, float(a), width)
            for i, a in enumerate(grids[kind])
        )
    rows = run_bank(f"{label}Calibration", spec, cfg, cases, out / "calibration", build)

    # 各校正結果から閾値の区間を求める。
    thresholds = {}
    size = amplitudes.size
    for j, kind in enumerate(("short", "long", "psc")):
        group = rows[j * size : (j + 1) * size]
        fired = [
            r["during_spikes"] + (r["post_spikes"] if kind == "psc" else 0) > 0
            for r in group
        ]
        if group[0]["during_spikes"] + group[0]["post_spikes"]:
            raise RuntimeError(f"{label}: 無入力自発発火。校正結果を確認してください")
        thresholds[kind] = threshold_bracket(grids[kind], fired)
    save_json(out / "thresholds.json", thresholds)
    fig, ax = plt.subplots(figsize=(7, 4))
    long_rows = rows[size : 2 * size]
    ax.plot(amplitudes, [r["late_stimulus_rate_hz"] for r in long_rows], "o-")
    ax.set(
        xlabel=f"Input amplitude ({'nA' if label == 'LIF' else 'model units'})",
        ylabel="Late step firing rate (Hz)",
        title=label,
    )
    fig.tight_layout()
    fig.savefig(out / "fi_curve.png", dpi=150)
    plt.close(fig)
    return thresholds


def recovery(label, spec, cfg, thresholds, out, build):
    amp = cfg["stimulus_factor"] * thresholds["psc"]["upper_firing"]
    width = 20 * cfg["tau_syn_ms"]
    cases = [Case("single", "exp", amp, width)]
    cases.extend(
        Case(f"pair_{gap}", "exp", amp, width, second_gap_ms=gap)
        for gap in cfg["recovery_gaps_ms"]
    )
    rows = run_bank(f"{label}Recovery", spec, cfg, cases, out / "recovery", build)

    # 二連刺激の結果を単発刺激の対照と比較する。
    spike_data = np.loadtxt(
        out / "recovery/spikes.csv", delimiter=",", skiprows=1, ndmin=2
    )
    single = spike_data[spike_data[:, 1] == 0, 0]
    for row in rows:
        onset = cfg["warmup_ms"] + max(0, row["second_gap_ms"])
        control_count = int(
            np.count_nonzero((single >= onset) & (single < row["observation_end_ms"]))
        )
        row["single_control_spikes_after_second_onset"] = control_count
        row["extra_spikes_vs_single"] = row["spikes_after_second_onset"] - control_count
    save_table(out / "recovery/summary.csv", rows)


def sweep(label, spec, cfg, thresholds, out, build):
    amp = cfg["stimulus_factor"] * thresholds["short"]["upper_firing"]
    reference = thresholds["psc"]["upper_firing"]
    cases = [
        Case(
            f"w{factor}_d{delay}",
            "rect",
            amp,
            cfg["short_ms"],
            weight=factor * reference,
            delay_ms=matched_dendritic_delay(
                delay, cfg["dt_ms"], cfg.get("reference_dt_ms")
            ),
            weight_factor=factor,
        )
        for factor in cfg["weight_factors"]
        for delay in cfg["delays_ms"]
    ]
    rows = run_bank(f"{label}Autapses", spec, cfg, cases, out / "autapses", build)

    for row in rows:
        if row["weight_factor"] == 0 and row["tail_spikes"]:
            raise RuntimeError(
                f"{label}: 重み0対照が末尾まで発火。自己帰還の効果を判別できません"
            )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ("post_spikes", "last_post_latency_ms", "tail_spikes")):
        data = np.array([np.nan if r[metric] is None else r[metric] for r in rows])
        data = data.reshape(len(cfg["weight_factors"]), len(cfg["delays_ms"]))
        im = ax.imshow(data, origin="lower", aspect="auto")
        ax.set(
            xticks=range(len(cfg["delays_ms"])),
            xticklabels=cfg["delays_ms"],
            yticks=range(len(cfg["weight_factors"])),
            yticklabels=cfg["weight_factors"],
            xlabel="Dendritic delay (ms)",
            ylabel="Weight / PSC threshold",
            title=metric,
        )
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "autapse_map.png", dpi=150)
    plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/autapse_comparison.yaml"
    )
    parser.add_argument(
        "--stage", choices=("baseline", "all", "verify"), default="baseline"
    )
    parser.add_argument(
        "--reference-run", type=Path, help="verifyで使用する完了済みall実験"
    )
    parser.add_argument("--dt-ms", type=float, help="時間刻みの上書き")
    parser.add_argument("--post-ms", type=float, help="刺激後観測時間の上書き")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs/autapse_comparison"
    )
    parser.add_argument(
        "--genn-code-dir", type=Path, default=ROOT / "genn_code/autapse_comparison"
    )
    args = parser.parse_args()
    if args.stage == "verify":
        if args.reference_run is None:
            parser.error("verifyには--reference-runが必要です")
        previous = json.loads((args.reference_run / "run.json").read_text())
        if previous["status"] != "complete" or previous["stage"] != "all":
            parser.error("reference-runには完了済みall実験を指定してください")
        cfg = yaml.safe_load((args.reference_run / "comparison.yaml").read_text())
        cfg["reference_dt_ms"] = cfg["dt_ms"]
        cfg["dt_ms"] = args.dt_ms if args.dt_ms is not None else cfg["dt_ms"] / 2
        cfg["post_ms"] = (
            args.post_ms if args.post_ms is not None else cfg["post_ms"] * 5
        )
    else:
        if args.reference_run is not None:
            parser.error("reference-runはverify専用です")
        cfg = yaml.safe_load(args.config.read_text())
        if args.dt_ms is not None:
            cfg["dt_ms"] = args.dt_ms
        if args.post_ms is not None:
            cfg["post_ms"] = args.post_ms
    validate_config(cfg)

    for delay in cfg["delays_ms"]:
        matched_dendritic_delay(delay, cfg["dt_ms"], cfg.get("reference_dt_ms"))
    run = args.out_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run.mkdir(parents=True)
    (run / "comparison.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    )
    run_info = {
        "stage": args.stage,
        "status": "running",
        "reference_run": (
            str(args.reference_run.resolve()) if args.reference_run else None
        ),
    }
    save_json(run / "run.json", run_info)
    print(f"出力先: {run}", flush=True)

    try:
        for label, spec in cfg["models"].items():
            model_dir = run / label
            model_dir.mkdir()
            build = args.genn_code_dir / run.name / label
            if args.stage == "verify":
                thresholds = json.loads(
                    (args.reference_run / label / "thresholds.json").read_text()
                )
                save_json(model_dir / "thresholds.json", thresholds)
            else:
                thresholds = calibrate(label, spec, cfg, model_dir, build)
                recovery(label, spec, cfg, thresholds, model_dir, build)
            if args.stage in ("all", "verify"):
                sweep(label, spec, cfg, thresholds, model_dir, build)
    except Exception as exc:
        save_json(run / "run.json", {**run_info, "status": "failed", "error": str(exc)})
        raise
    save_json(run / "run.json", {**run_info, "status": "complete"})
    print(f"比較実験を完了しました: {run}", flush=True)


if __name__ == "__main__":
    main()
