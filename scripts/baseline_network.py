"""PQN基準ネットワークの配線保存、校正、単条件実行、走査、再検証。"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.core.network_snapshot import load_snapshot
from src.utils.experiments.baseline_network import (
    BaselineSettings,
    dump_csv,
    dump_json,
    make_snapshot,
    run_case,
)


def run_logged(cfg, snapshot, out, build, **kwargs):
    row = run_case(cfg, snapshot, out, build, **kwargs)
    print(
        f"{out.name}: post={row['post_spikes']}, tail={row['tail_spikes']}, "
        f"{row['activity_status']}",
        flush=True,
    )
    return row


def plot_sweep(rows, output):
    stimulated = [r for r in rows if not r["no_stimulus"]]
    if not stimulated:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for inhibition in sorted({r["inhibition_factor"] for r in stimulated}):
        group = [r for r in stimulated if r["inhibition_factor"] == inhibition]
        scales = sorted({r["multiplier"] for r in group})
        for ax, key in zip(axes, ("post_rate_hz", "tail_spikes")):
            means = [
                np.mean([r[key] for r in group if r["multiplier"] == x]) for x in scales
            ]
            ax.plot(scales, means, "o-", label=f"I factor={inhibition}")
            for row in group:
                ax.scatter(row["multiplier"], row[key], s=10, alpha=0.25)
            ax.set(xlabel="Coupling multiplier", ylabel=key)
            ax.legend()
    fig.suptitle("Finite trials only; see numerical_failures.json for excluded trials")
    fig.tight_layout()
    fig.savefig(output / "sweep.png", dpi=150)
    plt.close(fig)


def calibrate(cfg, output, build):
    amplitudes = np.array(
        [0, -0.2, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8]
    )
    base = cfg.model_dump()
    base.update(
        n=2 * amplitudes.size,
        n_exc=amplitudes.size,
        probability=0.0,
        wiring_seeds=[1],
        stimulus_seeds=[101],
        stimulus_fraction=1.0,
        sample_ms=cfg.dt_ms,
    )
    isolated = BaselineSettings.model_validate(base)
    snapshot = output / "isolated.npz"
    make_snapshot(isolated, 1, snapshot)
    rows = []
    for width in (20.0, 500.0):
        trial = BaselineSettings.model_validate({**base, "pulse_ms": width})
        directory = output / f"pulse_{int(width)}ms"
        run_logged(
            trial,
            snapshot,
            directory,
            build / directory.name,
            multiplier=0,
            amplitude_vector=np.tile(amplitudes, 2),
        )
        data = np.load(directory / "spikes.npz")
        times, ids = data["times_ms"], data["ids"]
        for cell in range(isolated.n):
            own = times[ids == cell]
            rows.append(
                dict(
                    mode="RSexci" if cell < amplitudes.size else "RSinhi",
                    pulse_ms=width,
                    amplitude=float(amplitudes[cell % amplitudes.size]),
                    pre_spikes=int(np.count_nonzero(own < trial.warmup_ms)),
                    during_spikes=int(
                        np.count_nonzero(
                            (own >= trial.warmup_ms) & (own < trial.warmup_ms + width)
                        )
                    ),
                    post_spikes=int(np.count_nonzero(own >= trial.warmup_ms + width)),
                )
            )
    # 発火数だけでは持続的な脱分極と反復発火を区別できないため、波形も残す。
    with np.load(output / "pulse_500ms" / "traces.npz") as traces:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for group, ax in enumerate(axes):
            for index in (7, 9, 11):
                ax.plot(
                    traces["time_end_ms"],
                    traces["V"][:, group * amplitudes.size + index],
                    label=f"I={amplitudes[index]}",
                    linewidth=0.8,
                )
            ax.set_ylabel("V (model units)")
            ax.set_title(("RSexci", "RSinhi")[group])
            ax.legend()
        axes[-1].set_xlabel("Time (ms)")
        fig.tight_layout()
        fig.savefig(output / "calibration_waveforms.png", dpi=150)
        plt.close(fig)

    dump_csv(output / "calibration.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, width in zip(axes, (20, 500)):
        for mode in ("RSexci", "RSinhi"):
            group = sorted(
                [r for r in rows if r["mode"] == mode and r["pulse_ms"] == width],
                key=lambda r: r["amplitude"],
            )
            ax.plot(
                [r["amplitude"] for r in group],
                [r["during_spikes"] * 1000 / width for r in group],
                "o-",
                label=mode,
            )
        ax.set(
            xlabel="Current (model units)",
            ylabel="Rate (Hz)",
            title=f"{width} ms pulse",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "calibration.png", dpi=150)
    plt.close(fig)
    if any(
        r["pre_spikes"]
        or (r["amplitude"] == 0 and (r["during_spikes"] or r["post_spikes"]))
        for r in rows
    ):
        raise RuntimeError(
            "校正で無入力発火を検出しました。静止条件を再検討してください"
        )
    # 同じ外部入力に、RSinhiからの負の指数電流を加えた対照を比較する。
    pair_values = cfg.model_dump()
    pair_values.update(
        n=2,
        n_exc=1,
        probability=1.0,
        pulse_ms=500.0,
        wiring_seeds=[1],
        stimulus_seeds=[101],
        stimulus_fraction=1.0,
        weights={"ee": 0, "ei": 0, "ie": -0.4, "ii": 0},
    )
    pair_cfg = BaselineSettings.model_validate(pair_values)
    pair_snapshot = output / "inhibition_pair.npz"
    make_snapshot(pair_cfg, 1, pair_snapshot)
    comparisons = []
    for multiplier in (0, 1):
        directory = output / f"inhibition_{multiplier}"
        run_logged(
            pair_cfg,
            pair_snapshot,
            directory,
            build / directory.name,
            multiplier=multiplier,
            amplitude_vector=np.array([0.4, 0.4]),
        )
        with np.load(directory / "spikes.npz") as spikes:
            during = (spikes["times_ms"] >= pair_cfg.warmup_ms) & (
                spikes["times_ms"] < pair_cfg.warmup_ms + pair_cfg.pulse_ms
            )
            exc_times = spikes["times_ms"][during & (spikes["ids"] == 0)]
            comparisons.append(
                {
                    "ie_weight": -0.4 * multiplier,
                    "exc_during_spikes": int(exc_times.size),
                    "exc_first_latency_ms": (
                        float(exc_times[0] - pair_cfg.warmup_ms)
                        if exc_times.size
                        else None
                    ),
                    "inh_during_spikes": int(
                        np.count_nonzero(during & (spikes["ids"] == 1))
                    ),
                }
            )
    dump_csv(output / "inhibition_comparison.csv", comparisons)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run", "sweep", "calibrate", "verify")
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/baseline_network.yaml"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs/baseline_network"
    )
    parser.add_argument(
        "--code-dir", type=Path, default=ROOT / "genn_code/baseline_network"
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--reference-run", type=Path, help="verifyの基準となる単条件ディレクトリ"
    )
    parser.add_argument("--n", type=int, help="実装確認用サイズ。E/I比は設定から維持")
    parser.add_argument("--dt-ms", type=float)
    parser.add_argument("--post-ms", type=float)
    parser.add_argument("--warmup-ms", type=float)
    parser.add_argument("--stimulus-amplitude", type=float)
    parser.add_argument("--stimulus-fraction", type=float)
    parser.add_argument("--multiplier", type=float)
    parser.add_argument("--inhibition-factor", type=float)
    parser.add_argument("--stimulus-seed", type=int)
    parser.add_argument("--no-stimulus", action="store_true")
    args = parser.parse_args()
    if args.command not in ("run", "verify") and (
        args.multiplier is not None
        or args.inhibition_factor is not None
        or args.stimulus_seed is not None
        or args.no_stimulus
    ):
        parser.error(
            "単条件オプションはrun/verify専用です。sweepの走査値はYAMLへ指定してください"
        )
    if args.command == "calibrate" and args.snapshot is not None:
        parser.error(
            "calibrateは独立細胞の専用配線を生成するためsnapshotは指定できません"
        )
    ref = None
    if args.command == "verify":
        if (
            args.reference_run is None
            or args.n is not None
            or args.snapshot is not None
        ):
            parser.error(
                "verifyにはreference-runを指定し、n・snapshotは変更しないでください"
            )
        if (
            json.loads((args.reference_run / "status.json").read_text())["status"]
            != "complete"
        ):
            parser.error("完了した単条件実験を指定してください")
        values = yaml.safe_load((args.reference_run / "settings.yaml").read_text())
        ref = json.loads((args.reference_run / "metadata.json").read_text())
        values["dt_ms"] /= 2
        values["post_ms"] *= 2
    else:
        if args.reference_run is not None:
            parser.error("reference-runはverify専用です")
        values = yaml.safe_load(args.config.read_text())
    if args.n is not None:
        values["n_exc"] = int(np.floor(args.n * values["n_exc"] / values["n"] + 0.5))
        values["n"] = args.n
    for key in (
        "dt_ms",
        "post_ms",
        "warmup_ms",
        "stimulus_amplitude",
        "stimulus_fraction",
    ):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    cfg = BaselineSettings.model_validate(values)
    output = args.out_dir / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    build = args.code_dir / output.name
    print(f"出力先: {output}", flush=True)
    (output / "settings.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(), allow_unicode=True, sort_keys=False)
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    manifest = dict(
        command=args.command,
        argv=sys.argv[1:],
        git_commit=commit.stdout.strip(),
        reference_run=str(args.reference_run) if ref else None,
        status="running",
    )
    dump_json(output / "manifest.json", manifest)
    rows, failures = [], []
    try:
        if args.command == "calibrate":
            calibrate(cfg, output, build)
        else:
            snapshots = []
            supplied = Path(ref["snapshot_path"]) if ref else args.snapshot
            if supplied:
                arrays, meta, fingerprint = load_snapshot(supplied)
                if ref and fingerprint != ref["snapshot_fingerprint"]:
                    raise ValueError("基準実験の配線と一致しません")
                destination = output / f"wiring_{meta['wiring_seed']}.npz"
                shutil.copyfile(supplied, destination)
                snapshots.append(destination)
            else:
                seeds = (
                    cfg.wiring_seeds
                    if args.command in ("prepare", "sweep")
                    else cfg.wiring_seeds[:1]
                )
                for seed in seeds:
                    snapshot = output / f"wiring_{seed}.npz"
                    make_snapshot(cfg, seed, snapshot)
                    snapshots.append(snapshot)
            conditions = []
            for snapshot in snapshots:
                _, meta, _ = load_snapshot(snapshot)
                if args.command == "sweep":
                    for inhibition in cfg.inhibition_factors:
                        for multiplier in cfg.multipliers:
                            for seed in cfg.stimulus_seeds:
                                conditions.append(
                                    (snapshot, multiplier, inhibition, seed, False)
                                )
                        conditions.append(
                            (
                                snapshot,
                                max(cfg.multipliers),
                                inhibition,
                                meta["stimulus_seeds"][0],
                                True,
                            )
                        )
                elif args.command in ("run", "verify"):
                    defaults = ref or {
                        "multiplier": 1.0,
                        "inhibition_factor": 1.0,
                        "stimulus_seed": meta["stimulus_seeds"][0],
                        "no_stimulus": False,
                    }
                    multiplier = (
                        defaults["multiplier"]
                        if args.multiplier is None
                        else args.multiplier
                    )
                    inhibition = (
                        defaults["inhibition_factor"]
                        if args.inhibition_factor is None
                        else args.inhibition_factor
                    )
                    seed = (
                        defaults["stimulus_seed"]
                        if args.stimulus_seed is None
                        else args.stimulus_seed
                    )
                    no_stimulus = args.no_stimulus or defaults["no_stimulus"]
                    conditions.append(
                        (snapshot, multiplier, inhibition, seed, no_stimulus)
                    )
            for index, (
                snapshot,
                multiplier,
                inhibition,
                seed,
                no_stimulus,
            ) in enumerate(conditions):
                directory = output / f"case_{index:04d}"
                try:
                    rows.append(
                        run_logged(
                            cfg,
                            snapshot,
                            directory,
                            build / directory.name,
                            multiplier=multiplier,
                            inhibition_factor=inhibition,
                            stimulus_seed=seed,
                            no_stimulus=no_stimulus,
                        )
                    )
                except FloatingPointError as exc:
                    failures.append(dict(run_dir=str(directory), error=str(exc)))
                    print(
                        f"{directory.name}: 数値不安定（保存して次条件へ）", flush=True
                    )
                dump_csv(output / "summary.csv", rows)
                dump_json(output / "numerical_failures.json", failures)
            if rows:
                plot_sweep(rows, output)
    except Exception as exc:
        dump_json(
            output / "manifest.json",
            {**manifest, "status": "failed", "error": str(exc)},
        )
        raise
    dump_json(
        output / "manifest.json",
        {
            **manifest,
            "status": (
                "complete" if not failures else "complete_with_numerical_failures"
            ),
            "completed_cases": 4 if args.command == "calibrate" else len(rows),
            "failed_cases": len(failures),
        },
    )
    print(f"完了: {output}", flush=True)


if __name__ == "__main__":
    main()
