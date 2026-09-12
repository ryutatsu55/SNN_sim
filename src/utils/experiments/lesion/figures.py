"""損傷実験に固有の図。

汎用の描画 (`src/utils/plotting/`) では言えないことだけをここに置く —— 具体的には
「切断の瞬間をまたいで重みがどう動いたか」。`akita_soc/fig2c.py` の左列と同じ
「実在するシナプス 1 本 1 本の軌跡を薄い線で束ねる」形を踏襲するが、損傷実験には
あちらに無い事情が 2 つある:

1. **baseline と post でシナプス本数が違う。** baseline は Phase 1 (切断前・全シナプス)、
   post は Phase 2 (切断後)。同じ図に載せるには、生き残ったシナプスに揃えて引き当てる
   必要がある (`restore.align_subset_to_coo`)。
2. **時間軸が切断からの相対時間で、負の領域を持つ。** baseline は切断より前に測った値。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.core.output_manager import CONNECTIVITY_NAME, data_dir  # noqa: E402
from src.utils.analysis.spikes import firing_rates  # noqa: E402
from src.utils.analysis.weights import (BLOCK_ORDER, block_masks,  # noqa: E402
                                        excitatory_flags)
from src.utils.experiments.lesion import runio  # noqa: E402
from src.utils.experiments.lesion.restore import align_subset_to_coo  # noqa: E402
from src.utils.plotting.common import save_figure  # noqa: E402

# fig2c と同じ。1 本ずつの線は薄く重ねて「束」として読ませる。
LINE_ALPHA = 0.02
LINE_WIDTH = 0.5


def load_weight_trajectories(run_dir: Path, layout):
    """probe 群から、**生き残ったシナプス**の重み軌跡を E/I ブロック別に集める。

    Returns:
        (hours, trajectories, phases)
        hours       : (T,) 切断からの経過 [h] (baseline は負)
        trajectories: {"W_EE": (T, n_EE), ...}
        phases      : (T,) "baseline" / "post"
    """
    source = data_dir(run_dir)
    probes = runio.discover_probes(source, runio.WEIGHTS)
    if not probes:
        print(f"警告: {source} に weights_p*.npz が見つかりません。")
        return None, None, None

    connectivity = np.load(source / CONNECTIVITY_NAME, allow_pickle=False)
    row = np.asarray(connectivity["row"], dtype=np.int64)
    col = np.asarray(connectivity["col"], dtype=np.int64)
    n_cols = int(np.asarray(connectivity["shape"])[1])

    masks = block_masks(row, col, excitatory_flags(layout, layout.total_neurons))

    # 切断前の probe は Phase 1 の全シナプスぶんあるので、生き残った分へ引き当てる。
    pre_path = source / runio.PRE_WEIGHTS_NAME
    pre = np.load(pre_path, allow_pickle=False) if pre_path.exists() else None

    times, phases = [], []
    traj = {name: [] for name in BLOCK_ORDER}
    schedule = _probe_schedule(source)
    for probe in probes:
        values = np.asarray(np.load(probe.path, allow_pickle=False)["data"], dtype=np.float64)
        if values.size != row.size:
            # 切断前 (Phase 1) の probe。weights_at_cut.npz の row/col で引き当てる。
            if pre is None or values.size != np.asarray(pre["row"]).size:
                print(f"警告: probe {probe.index} の本数 {values.size} を"
                      f" 切断後の {row.size} 本へ引き当てられません。飛ばします。")
                continue
            values = align_subset_to_coo(pre["row"], pre["col"], values, row, col, n_cols)
        entry = schedule.get(probe.index, {})
        times.append(float(entry.get("hours_since_cut", probe.index)))
        phases.append(str(entry.get("phase", "")))
        for name in BLOCK_ORDER:
            traj[name].append(values[masks[name]])

    if not times:
        return None, None, None
    order = np.argsort(times)
    hours = np.asarray(times)[order]
    trajectories = {f"W_{name}": np.asarray(traj[name])[order] for name in BLOCK_ORDER}
    return hours, trajectories, np.asarray(phases)[order]


def _probe_schedule(source: Path) -> dict[int, dict]:
    """probes.csv を {index: 行} で読む。時刻はファイル名でなくここが持つ。"""
    import csv
    path = source / runio.PROBES_NAME
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        return {int(row["index"]): row for row in csv.DictReader(handle)}


def plot_weight_trajectories(run_dir, layout, output_dir=None, wmax: float = 1.0) -> None:
    """重み軌跡を E/I ブロック別に 4 段で描く (fig2c 左列の損傷実験版)。

    薄い線の束だけだと probe 数が少ないときに読めないので、**ブロック平均を重ねる**。
    切断の瞬間には縦の破線を引く。
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    hours, trajectories, phases = load_weight_trajectories(run_dir, layout)
    if hours is None:
        print("エラー: 重み軌跡を集められませんでした。")
        return

    fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
    for index, name in enumerate(BLOCK_ORDER):
        ax = axes[index]
        values = trajectories[f"W_{name}"]
        if values.size:
            ax.plot(hours, values, color="black", alpha=LINE_ALPHA, linewidth=LINE_WIDTH)
            ax.plot(hours, values.mean(axis=1), color="tab:red", linewidth=1.6,
                    label=f"mean (n={values.shape[1]})")
            ax.legend(fontsize=7, loc="upper right")
        ax.axvline(0.0, color="tab:blue", linestyle="--", linewidth=1.2)
        ax.set_ylim(-0.05 * wmax, 1.05 * wmax)
        ax.set_yticks([0.0, 0.5 * wmax, wmax])
        ax.set_ylabel(f"W_{name}")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_title("Synaptic weights across the lesion\n"
                      "(surviving synapses only; dashed line = lesion)")
    axes[-1].set_xlabel("Time since lesion [h]")
    fig.tight_layout()
    save_figure(fig, output_dir / "weight_trajectories.png")
    print(f"グラフを {output_dir / 'weight_trajectories.png'} に保存しました。")


def plot_weight_distribution_shift(run_dir, layout, output_dir=None, bins: int = 60) -> None:
    """切断直前 / 直後 / 最終の重み**分布**を 1 枚に重ねる。

    軌跡図が「各シナプスがどう動いたか」を見せるのに対し、こちらは「集団としての形が
    どう変わったか」を見せる。回復が分布の平行移動なのか、両端への二極化なのかは
    軌跡の束からは読み取りにくい。
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    hours, trajectories, _phases = load_weight_trajectories(run_dir, layout)
    if hours is None:
        return

    # ラベルは ASCII にする。既定フォント (DejaVu Sans) に CJK が無く、豆腐になるため。
    picks = [(0, "before lesion")]
    if hours.size > 1:
        picks.append((1, "just after"))
    if hours.size > 2:
        picks.append((hours.size - 1, "final"))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, name in enumerate(BLOCK_ORDER):
        ax = axes.flat[index]
        values = trajectories[f"W_{name}"]
        if not values.size:
            ax.set_title(f"W_{name} (0 synapses)")
            continue
        for pick, label in picks:
            ax.hist(values[pick], bins=edges, histtype="step", linewidth=1.4,
                    label=f"{label} ({hours[pick]:+.2f} h)")
        ax.set_title(f"W_{name}  (n={values.shape[1]})")
        ax.set_xlabel("weight")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("Weight distribution before / after the lesion (surviving synapses)")
    fig.tight_layout()
    save_figure(fig, output_dir / "weight_distribution_shift.png")
    print(f"グラフを {output_dir / 'weight_distribution_shift.png'} に保存しました。")


def load_firing_rate_series(run_dir: Path, layout):
    """probe 群から、ニューロン別の発火レートを集める (fig2d の損傷実験版)。

    fig2d との違いは **窓幅を probe ごとに読む**こと。あちらは config の
    `record_window_ms` 1 つで割ればよいが、損傷実験では baseline と post で窓幅を
    変えられる (`--baseline-window-ms`)。全部を同じ幅で割ると、窓の違いが
    そのまま発火率の段差として現れてしまう。

    ニューロン数は切断で変わらない (消したのはシナプス) ので、重み軌跡と違って
    引き当ては要らない。

    Returns:
        (hours, rates, exc_ids, inh_ids)
        hours: (T,) 切断からの経過 [h] (baseline は負)
        rates: (T, N) Hz
    """
    source = data_dir(run_dir)
    probes = runio.discover_probes(source, runio.SPIKES)
    if not probes:
        print(f"警告: {source} に spikes_p*.npz が見つかりません。")
        return None, None, None, None

    schedule = _probe_schedule(source)
    total_neurons = layout.total_neurons
    polarity = layout.ids_by("polarity")

    times, rates = [], []
    for probe in probes:
        entry = schedule.get(probe.index, {})
        window_ms = float(entry.get("window_ms", 0.0) or 0.0)
        if window_ms <= 0.0:
            print(f"警告: probe {probe.index} の window_ms が不正です。飛ばします。")
            continue
        with np.load(probe.path, allow_pickle=False) as data:
            ids = data["ids"]
        times.append(float(entry.get("hours_since_cut", probe.index)))
        rates.append(firing_rates(ids, total_neurons, window_ms))

    if not times:
        return None, None, None, None
    order = np.argsort(times)
    return (np.asarray(times)[order], np.asarray(rates)[order],
            polarity.get("excitatory"), polarity.get("inhibitory"))


def plot_firing_rate_scatter(run_dir, layout, output_dir=None) -> None:
    """ニューロン別の発火レート推移を散布図で描く (fig2d の損傷実験版)。

    興奮性=赤 / 抑制性=青。切断の瞬間に縦の破線を引き、E/I それぞれの平均を重ねる
    —— 点が N×T 個あると個々の軌跡は追えないので、群としての落ち込みと戻りは
    平均線で読ませる。
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    hours, rates, exc_ids, inh_ids = load_firing_rate_series(run_dir, layout)
    if hours is None:
        print("エラー: 発火レートデータを集められませんでした。")
        return

    n_neurons = rates.shape[1]
    t_grid = np.repeat(hours[:, None], n_neurons, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for ids, color, label in ((inh_ids, "blue", "Inhibitory"), (exc_ids, "red", "Excitatory")):
        if ids is None or np.asarray(ids).size == 0:
            continue
        ids = np.asarray(ids)
        ids = ids[ids < n_neurons]
        ax.scatter(t_grid[:, ids].ravel(), rates[:, ids].ravel(),
                   s=6, c=color, alpha=0.35, edgecolors="none", label=label)
        ax.plot(hours, rates[:, ids].mean(axis=1), color=color, linewidth=1.8,
                label=f"{label} mean")

    ax.axvline(0.0, color="tab:blue", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Time since lesion [h]")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Per-neuron firing rate across the lesion\n(dashed line = lesion)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", markerscale=2.0, fontsize=8)
    span = float(hours.max() - hours.min())
    pad = 0.05 * span if span > 0 else 0.5
    ax.set_xlim(hours.min() - pad, hours.max() + pad)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_figure(fig, output_dir / "firing_rate_scatter.png")
    print(f"グラフを {output_dir / 'firing_rate_scatter.png'} に保存しました。")
