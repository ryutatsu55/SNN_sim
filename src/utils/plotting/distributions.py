"""**値の分布**の描画。「何本あるか」ではなく「どんな値が何本あるか」を見る図。

2 系統を持つ:

- 離散分布の log-log (アバランシェサイズ)。フィットそのものは
  `src.utils.analysis.powerlaw.fit_distribution_curves` が済ませており、ここは経験 PMF の
  散布と理論曲線の重ね描き、および凡例の体裁だけを担う。
- シナプス量のヒストグラム (重み・遅延・距離)。全体に加えて E/I ブロック別のパネルを
  添える。骨格は `_synapse_value_distribution` 1 つで、遅延も距離も重みもその薄い包み。
  値は COO から取るので、実在する結合の値だけが数えられる —— 結合の無い箇所の 0 が
  分布に山を作ることはない。

図はどれも `(view, out_path)` を取る (`src/utils/runview.py` の契約)。`ax` を取る
`draw_discrete_distribution` だけはプリミティブで、複数の図が同じパネルを使い回せる
ようにしてある (`area.py` の `draw_area` と同じ作法)。
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
from src.utils.analysis.avalanche import split_avalanches
from src.utils.plotting.common import BLOCK_COLORS, save_figure
from src.utils.runview import MissingData

# 見た目。**引数にしない** —— 変えたくなったらここを直す。
DPI = 200
BINS = 80
# 論文 (Ikeda-Akita-Takahashi 2023) Fig.2 の軸。図を並べて比べるための固定値。
AVALANCHE_XLIM = (1.0, 1000.0)
AVALANCHE_YLIM = (1e-5, 1.0)


def draw_discrete_distribution(
    ax,
    fit: DistributionFit,
    xlabel: str,
    reference_slope: float | None = None,
) -> None:
    """経験 PMF に power-law / exponential の最尤フィットを重ねて log-log で描く。

    Args:
        ax: 描画先。図の生成と保存は呼び出し側の責任。
        fit: `fit_distribution_curves` の結果。
        xlabel: x 軸ラベル。
        reference_slope: 指定すると、その傾きの参照直線を経験分布の先頭に合わせて重ねる。
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


def avalanche_distribution(window, out_path: Path) -> None:
    """記録窓のスパイク列からアバランチを切り出し、サイズ分布を log-log で描く。

    べき乗フィットの打ち切りは**系のサイズ N**。smax はデータの切り取りではなく
    モデルの正規化台 (p(s) = s^-α / Σ_{k=1}^{smax} k^-α) なので、観測サイズが smax を
    超えなくても smax を変えれば α は動く。
    """
    smax = int(window.config.simulation.N)
    sizes = split_avalanches(np.asarray(window.spikes().times, dtype=np.float64)).sizes
    fit = fit_distribution_curves(sizes, fit_max=smax)

    fig, ax = plt.subplots(figsize=(5, 4))
    draw_discrete_distribution(ax, fit, xlabel="Avalanche size")
    if fit.support.size == 0:
        # データが無くても log 軸の枠だけは描いておく。
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(f"Avalanche distribution {window.hour:g} h")
    ax.set_xlim(*AVALANCHE_XLIM)
    ax.set_ylim(*AVALANCHE_YLIM)
    save_figure(fig, out_path, dpi=DPI)


def _synapse_value_distribution(view, values: np.ndarray, out_path: Path, *,
                                xlabel: str, title: str, unit: str = "") -> None:
    """COO 上の per-synapse 量のヒストグラム (左: 全体 / 右: E/I ブロック別)。

    遅延・距離・重みの共通の骨格。右パネルは左と**同じビン境界**を使うので、
    ブロック別の山が全体のどこに乗っているか読める。

    Args:
        view: 契約の view。row/col と E/I の分類をここから取る。
        values: 各シナプスの値 (1D, wiring と index 整合)
        xlabel: 横軸ラベル (単位を含めて呼び出し側が決める)
        title: 図全体のタイトル
        unit: 左パネルの mean/max に添える単位。空なら数値だけ。
    """
    wiring = view.wiring()
    if wiring.num_synapses == 0:
        raise MissingData("synapses", "結合が 1 本もありません")
    values = np.asarray(values, dtype=np.float64)
    masks = block_masks(wiring.row, wiring.col,
                        excitatory_flags(view.layout, view.total_neurons))
    suffix = f" {unit}" if unit else ""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(values, bins=BINS, color="black")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Number of synapses")
    if values.size:
        axes[0].set_title(f"All synapses (n={values.size})\n"
                          f"mean={values.mean():.2f}{suffix}, max={values.max():.2f}{suffix}")
    else:
        axes[0].set_title("All synapses (empty)")

    edges = np.histogram_bin_edges(values, bins=BINS) if values.size \
        else np.linspace(0, 1, BINS)
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
    save_figure(fig, out_path, dpi=DPI)


def delay_distribution(view, out_path: Path) -> None:
    """実在する結合上の伝播遅延のヒストグラム (全体 + E/I ブロック別)。

    遅延を持つのは build 直後の COO だけ (記録窓の `Coo.delays` は None)。
    """
    delays = view.coo().delays
    if delays is None:
        raise MissingData("delays", "この view は遅延を持ちません")
    _synapse_value_distribution(view, delays, out_path,
                                xlabel="Delay [ms]", title="Delay distribution", unit="ms")


def distance_distribution(view, out_path: Path) -> None:
    """実在する結合の**長さ**のヒストグラム (全体 + E/I ブロック別)。

    `delay: distance_based` なら遅延の図と相似形になる。ずれたときは、遅延だけ頭打ち
    なら `max_delay` の clip、距離だけ広がっているなら伝導速度の設定。

    `matrices.empirical_connection_probability` とは分母が違う。あちらは確率
    (その距離のペアのうち何割が繋がったか)、こちらは件数。
    """
    wiring = view.wiring()
    _synapse_value_distribution(
        view, synapse_distances(view.coords(), wiring.row, wiring.col), out_path,
        xlabel="Distance [um]", title="Synapse distance distribution", unit="um")


def weight_distribution(view, out_path: Path) -> None:
    """その時点の重み分布 (全体 + E/I ブロック別)。時間発展は `matrices.weight_panel`。"""
    _synapse_value_distribution(view, view.weights(), out_path,
                                xlabel="Weight", title="Weight distribution")
