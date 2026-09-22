"""シナプス量 (重み・遅延・距離) のヒストグラム。

全体に加えて E/I ブロック別のパネルを添える。骨格は `synapse_value_distribution`
1 つで、遅延も距離もその薄い包み。入力は COO (row, col と index 整合の 1D 配列) なので、
実在する結合の値だけが数えられる —— 結合の無い箇所の 0 が分布に山を作ることはない。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.utils.analysis.weights import (
    BLOCK_ORDER,
    block_masks,
    excitatory_flags,
    synapse_distances,
)
from src.utils.runview import MissingData
from scripts.lesion.figures import save
from scripts.lesion.figures.style import BLOCK_COLORS


# 保存時の解像度。**引数にしない** —— 変えたくなったらここを直す。
DPI = 200
BINS = 80
HIST_FIGSIZE = (11, 4)
WEIGHT_TITLE = "Weight distribution over time"


def _value_distribution(built, values: np.ndarray, out_path: Path, *,
                        xlabel: str, title: str, unit: str = "") -> None:
    """COO 上の per-synapse 量のヒストグラム (左: 全体 / 右: E/I ブロック別)。

    遅延・距離・重みの共通の骨格。右パネルは左と**同じビン境界**を使うので、
    ブロック別の山が全体のどこに乗っているか読める。

    Args:
        built: 契約の `Built`。row/col と E/I の分類をここから取る。
        values: 各シナプスの値 (1D, wiring と index 整合)
        xlabel: 横軸ラベル (単位を含めて呼び出し側が決める)
        title: 図全体のタイトル
        unit: 左パネルの mean/max に添える単位。空なら数値だけ。
    """
    wiring = built.wiring()
    if wiring.num_synapses == 0:
        raise MissingData("synapses", "結合が 1 本もありません")
    values = np.asarray(values, dtype=np.float64)
    masks = block_masks(wiring.row, wiring.col,
                        excitatory_flags(built.layout, built.total_neurons))
    suffix = f" {unit}" if unit else ""

    fig, axes = plt.subplots(1, 2, figsize=HIST_FIGSIZE)

    axes[0].hist(values, bins=BINS, color="black")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Number of synapses")
    if values.size:
        axes[0].set_title(f"All synapses (n={values.size})\n"
                          f"mean={values.mean():.2f}{suffix}, max={values.max():.2f}{suffix}")
    else:
        axes[0].set_title("All synapses (empty)")

    edges = np.histogram_bin_edges(values, bins=BINS) if values.size else np.linspace(0, 1, BINS)
    drawn = 0
    for name in BLOCK_ORDER:
        block = values[masks[name]]
        if block.size:
            axes[1].hist(block, bins=edges, histtype="step", lw=1.4,
                         color=BLOCK_COLORS[name], label=f"{name} (n={block.size})")
            drawn += 1
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Number of synapses")
    axes[1].set_title("By connection type")
    if drawn:
        # 1 本も描いていないときに legend() を呼ぶと matplotlib が警告を出すだけなので黙る。
        axes[1].legend(fontsize=7)

    fig.suptitle(title)
    save(fig, out_path, dpi=DPI)

def delay_distribution(built, out_path: Path) -> None:
    """実在する結合上の伝播遅延のヒストグラム (全体 + E/I ブロック別)。

    結合が無い箇所は行列上 0 で埋まるため、必ず COO (= 実結合のみ) を渡すこと。
    """
    _value_distribution(
        built, built.coo().delays, out_path,
        xlabel="Delay [ms]", title="Delay distribution", unit="ms",
    )

def distance_distribution(built, out_path: Path) -> None:
    """実在する結合の**長さ**のヒストグラム (全体 + E/I ブロック別)。

    `delay: distance_based` なら遅延の図と相似形になる。ずれたときは、遅延だけ頭打ち
    なら `max_delay` の clip、距離だけ広がっているなら伝導速度の設定。

    `empirical_connection_probability` とは分母が違う。あちらは確率
    (その距離のペアのうち何割が繋がったか)、こちらは件数。
    """
    wiring = built.wiring()
    _value_distribution(
        built, synapse_distances(built.coords(), wiring.row, wiring.col), out_path,
        xlabel="Distance [um]", title="Synapse distance distribution", unit="um",
    )

def weight_distribution(built, out_path: Path) -> None:
    """その時点の重み分布を全体 + E/I ブロック別のパネルで描く。

    時間発展は `weight_matrix.py` のパネル図と `fig2c.py` の軌跡が受け持つ。
    """
    wiring = built.wiring()
    if wiring.num_synapses == 0:
        raise MissingData("synapses", "結合が 1 本もありません")
    hours = [0.0]
    weight_arrays = [built.coo().weights]
    row, col, layout = wiring.row, wiring.col, built.layout
    total_neurons = built.total_neurons

    masks = block_masks(row, col, excitatory_flags(layout, total_neurons))

    num_panels = 1 + len(BLOCK_ORDER)
    columns = min(num_panels, 3)
    rows_needed = int(np.ceil(num_panels / columns))
    fig, axes = plt.subplots(rows_needed, columns,
                             figsize=(4.2 * columns, 3.4 * rows_needed), squeeze=False)
    flat_axes = axes.ravel()

    all_values = np.concatenate([np.asarray(w, dtype=np.float64) for w in weight_arrays]) \
        if weight_arrays else np.array([0.0, 1.0])
    edges = np.histogram_bin_edges(all_values, bins=BINS)
    colours = plt.cm.viridis(np.linspace(0, 0.9, max(len(hours), 1)))

    for hour, weights, colour in zip(hours, weight_arrays, colours):
        values = np.asarray(weights, dtype=np.float64)
        flat_axes[0].hist(values, bins=edges, histtype="step", lw=1.4,
                          color=colour, label=f"{hour:g} h")
    flat_axes[0].set_title("All synapses")
    flat_axes[0].set_xlabel("Weight")
    flat_axes[0].set_ylabel("Number of synapses")
    flat_axes[0].legend(fontsize=7)

    for panel, name in enumerate(BLOCK_ORDER, start=1):
        axis = flat_axes[panel]
        for hour, weights, colour in zip(hours, weight_arrays, colours):
            values = np.asarray(weights, dtype=np.float64)[masks[name]]
            if values.size:
                axis.hist(values, bins=edges, histtype="step", lw=1.3,
                          color=colour, label=f"{hour:g} h")
        axis.set_title(f"{name} synapses")
        axis.set_xlabel("Weight")
        axis.set_ylabel("Number of synapses")

    for unused in range(num_panels, flat_axes.size):
        flat_axes[unused].axis("off")

    fig.suptitle(WEIGHT_TITLE)
    save(fig, out_path, dpi=DPI)
