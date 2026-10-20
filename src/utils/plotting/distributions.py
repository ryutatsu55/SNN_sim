"""**値の分布**の描画。「何本あるか」ではなく「どんな値が何本あるか」を見る図。

2 系統を持つ:

- 離散分布の log-log (アバランシェサイズ・寿命など)。フィットそのものは
  `src.utils.analysis.powerlaw.fit_distribution_curves` が済ませており、ここは経験 PMF の
  散布と理論曲線の重ね描き、および凡例の体裁だけを担う。
- シナプス量のヒストグラム (重み・遅延・距離)。全体に加えて E/I ブロック別のパネルを
  添える。骨格は `plot_synapse_value_distribution` 1 つで、遅延も距離もその薄い包み。
  入力は COO (row, col と index 整合の 1D 配列) なので、実在する結合の値だけが数えられる
  — 結合の無い箇所の 0 が分布に山を作ることはない。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.analysis.powerlaw import DistributionFit, fit_distribution_curves
from src.utils.analysis.weights import (
    BLOCK_ORDER,
    block_masks,
    excitatory_flags,
    synapse_distances,
)
from src.utils.plotting.common import BLOCK_COLORS, save_figure


def plot_discrete_distribution(
    ax,
    fit: DistributionFit,
    xlabel: str,
    reference_slope: float | None = None,
) -> None:
    """経験 PMF に power-law / exponential の最尤フィットを重ねて log-log で描く。

    Args:
        ax: 描画先。図の生成と保存は呼び出し側の責任。
        fit: `fit_distribution_curves` の結果。
        xlabel: x 軸ラベル (「何の分布か」は実験によって違うので引数)。
        reference_slope: 指定すると、その傾きの参照直線を経験分布の先頭に合わせて重ねる
            (Beggs & Plenz のサイズ -3/2 / 寿命 -2 など)。None なら描かない。
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    if fit.support.size == 0:
        return

    ax.scatter(fit.support, fit.prob, s=12, color="black", label="empirical", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    if fit.powerlaw.size:
        ax.plot(fit.fit_support, fit.powerlaw, color="tab:red", lw=1.5, zorder=2,
                label=f"power-law MLE (α={fit.alpha:.2f})")
        ax.plot(fit.fit_support, fit.exponential, color="tab:blue", lw=1.2, ls=":", zorder=2,
                label=f"exponential MLE (λ={fit.lam:.3f})")
        if reference_slope is not None:
            # 参照傾きは経験分布の先頭に合わせて配置する。
            reference = fit.prob[0] * np.power(fit.fit_support, reference_slope)
            ax.plot(fit.fit_support, reference, color="gray", lw=1.2, ls="--", zorder=1,
                    label=f"reference slope {reference_slope}")

    ax.legend(fontsize=6.5, loc="lower left",
              title=f"fit slope={fit.slope_loglog:.2f}  LLR={fit.llr:.0f}",
              title_fontsize=6.5)


def plot_avalanche_distribution(
    sizes: np.ndarray,
    out_path: Path,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    smax: int | None = None,
    fit_smax: int = 100,
    reference_slope: float | None = None,
) -> None:
    """アバランシェサイズ分布を 1 枚の図として保存する (`plot_discrete_distribution` の図版)。"""
    fit = fit_distribution_curves(sizes, fit_max=fit_smax, support_max=smax)

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_discrete_distribution(ax, fit, xlabel="Avalanche size",
                               reference_slope=reference_slope)
    if fit.support.size == 0 and (xlim is not None or ylim is not None):
        # データが無くても軸範囲が指定されていれば log 軸の枠だけは描いておく。
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    save_figure(fig, out_path)


def plot_synapse_value_distribution(
    values: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    layout,
    total_neurons: int,
    out_path: Path,
    xlabel: str,
    title: str,
    unit: str = "",
    bins: int = 80,
) -> None:
    """COO 上の per-synapse 量のヒストグラム (左: 全体 / 右: E/I ブロック別)。

    「各シナプスに 1 つ値が付いている」ものなら何でも描ける汎用版。遅延・距離・重みは
    量が違うだけで見たい形は同じなので、図の骨格はここ 1 つに集約している。右パネルは
    左と**同じビン境界**を使うので、ブロック別の山が全体のどこに乗っているか読める。

    Args:
        values: 各シナプスの値 (1D, row/col と index 整合)
        row, col: 各シナプスの送信/受信グローバルID
        layout: NetworkLayout (E/I の分類は polarity 軸から取る)
        total_neurons: 全ニューロン数
        out_path: 出力ファイルパス
        xlabel: 横軸ラベル (単位を含めて呼び出し側が決める)
        title: 図全体のタイトル
        unit: 左パネルの mean/max に添える単位。空なら数値だけ。
        bins: ヒストグラムのビン数
    """
    values = np.asarray(values, dtype=np.float64)
    masks = block_masks(row, col, excitatory_flags(layout, total_neurons))
    suffix = f" {unit}" if unit else ""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(values, bins=bins, color="black")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Number of synapses")
    if values.size:
        axes[0].set_title(f"All synapses (n={values.size})\n"
                          f"mean={values.mean():.2f}{suffix}, max={values.max():.2f}{suffix}")
    else:
        axes[0].set_title("All synapses (empty)")

    edges = np.histogram_bin_edges(values, bins=bins) if values.size else np.linspace(0, 1, bins)
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
    save_figure(fig, out_path)


def plot_delay_distribution(
    delays_ms: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    layout,
    total_neurons: int,
    out_path: Path,
    title: str = "Delay distribution",
    bins: int = 80,
) -> None:
    """実在する結合上の伝播遅延のヒストグラム (全体 + E/I ブロック別)。

    結合が無い箇所は行列上 0 で埋まるため、必ず COO (= 実結合のみ) を渡すこと。
    """
    plot_synapse_value_distribution(
        delays_ms, row, col, layout, total_neurons, out_path,
        xlabel="Delay [ms]", title=title, unit="ms", bins=bins,
    )


def plot_distance_distribution(
    coords: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    layout,
    total_neurons: int,
    out_path: Path,
    title: str = "Synapse distance distribution",
    bins: int = 80,
) -> None:
    """実在する結合の**長さ**のヒストグラム (全体 + E/I ブロック別)。遅延版の距離版。

    距離依存の結合則では遅延が距離の一次関数なので、`delay: distance_based` ならこの図は
    遅延の図と相似形になる。両方を出す意味は、**一致しないとき**にどちらが原因かが分かる
    ことにある (遅延だけ頭打ち = `max_delay` の clip、距離だけ広がっている = 伝導速度の設定)。

    `plot_empirical_connection_probability` とは分母が違う。あちらは「その距離にある
    ペアのうち何割が繋がったか」(確率)、こちらは「実際に張られた結合が何本あるか」(件数)。
    ペアの数自体が距離とともに増えるので、確率が単調減少でも件数はピークを持つ。
    """
    plot_synapse_value_distribution(
        synapse_distances(coords, row, col), row, col, layout, total_neurons, out_path,
        xlabel="Distance [um]", title=title, unit="um", bins=bins,
    )


def plot_weight_distributions(
    hours: list[float],
    weight_arrays: list[np.ndarray],
    out_path: Path,
    row: np.ndarray | None = None,
    col: np.ndarray | None = None,
    layout=None,
    total_neurons: int | None = None,
    title: str = "Weight distribution over time",
    bins: int = 80,
) -> None:
    """各計測時刻の重み分布を 1 枚に重ね描きし、E/I ブロック別のパネルも添える。

    row/col/layout/total_neurons を渡すとブロック別パネルを描く。省略した場合は
    全体のヒストグラムのみ。
    """
    if len(hours) != len(weight_arrays):
        raise ValueError("hours と weight_arrays の長さが一致しません。")

    has_blocks = row is not None and col is not None and layout is not None and total_neurons
    masks = block_masks(row, col, excitatory_flags(layout, total_neurons)) if has_blocks else None

    num_panels = 1 + (len(BLOCK_ORDER) if has_blocks else 0)
    columns = min(num_panels, 3)
    rows_needed = int(np.ceil(num_panels / columns))
    fig, axes = plt.subplots(rows_needed, columns,
                             figsize=(4.2 * columns, 3.4 * rows_needed), squeeze=False)
    flat_axes = axes.ravel()

    all_values = np.concatenate([np.asarray(w, dtype=np.float64) for w in weight_arrays]) \
        if weight_arrays else np.array([0.0, 1.0])
    edges = np.histogram_bin_edges(all_values, bins=bins)
    colours = plt.cm.viridis(np.linspace(0, 0.9, max(len(hours), 1)))

    for hour, weights, colour in zip(hours, weight_arrays, colours):
        values = np.asarray(weights, dtype=np.float64)
        flat_axes[0].hist(values, bins=edges, histtype="step", lw=1.4,
                          color=colour, label=f"{hour:g} h")
    flat_axes[0].set_title("All synapses")
    flat_axes[0].set_xlabel("Weight")
    flat_axes[0].set_ylabel("Number of synapses")
    flat_axes[0].legend(fontsize=7)

    if has_blocks:
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

    fig.suptitle(title)
    save_figure(fig, out_path)
