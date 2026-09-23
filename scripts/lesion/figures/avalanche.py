"""アバランチサイズ分布の log-log 図。

フィットそのものは `src.utils.analysis.powerlaw.fit_distribution_curves` が済ませており、
ここは経験 PMF の散布・理論曲線の重ね描き・凡例の体裁だけを担う。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.utils.analysis.avalanche import split_avalanches
from src.utils.analysis.powerlaw import DistributionFit, fit_distribution_curves
from src.utils.analysis.criticality import DeltaCrFit, delta_cr_fit
from scripts.lesion.figures import save

# 論文 (Ikeda-Akita-Takahashi 2023) Fig.2 の軸。図を並べて比べるための固定値。
PAPER_XLIM = (1.0, 1000.0)
PAPER_YLIM = (1e-5, 1.0)
FIGSIZE = (5, 4)
DPI = 200


def draw_discrete_distribution(
    ax,
    fit: DistributionFit,
    xlabel: str,
    reference_slope: float | None = None,
    regression: DeltaCrFit | None = None,
) -> None:
    """経験 PMF に理論曲線を重ねて log-log で描く。

    重なる線は 2 系統ある。**別物なので線種も凡例も分けてある。**

    - 最尤フィット (赤 = power-law, 青の点線 = exponential)。LLR が比べているのはこの 2 本。
    - ΔCr の回帰直線 (緑の破線)。観測個数で重み付けた log-log 最小二乗で、
      ΔCr はこの直線と経験 PMF の**確率の差**を上振れ/下振れに分けて足したもの。

    Args:
        ax: 描画先。図の生成と保存は呼び出し側の責任。
        fit: `fit_distribution_curves` の結果。
        xlabel: x 軸ラベル。
        reference_slope: 指定すると、その傾きの参照直線を経験分布の先頭に合わせて重ねる。
        regression: `delta_cr_fit()` の結果。渡すと ΔCr の回帰直線を重ね、凡例に値を出す。
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

    # **傾きは凡例の各行が持つ。** 最尤の α と ΔCr 回帰の傾きは別物なので、
    # どちらとも読める "fit slope" をここに出さない。
    title = f"LLR={fit.llr:.0f}"
    if regression is not None:
        keep = regression.grid >= regression.smin
        ax.plot(regression.grid[keep], regression.fit[keep], color="tab:green", lw=1.2,
                ls="--", zorder=2,
                label=f"ΔCr regression (slope={regression.slope:.2f})")
        title += f"  ΔCr={regression.delta_cr:+.3f}"

    ax.legend(fontsize=6.5, loc="lower left", title=title, title_fontsize=6.5)

def avalanche_distribution(window, out_path: Path) -> None:
    """記録窓のスパイク列からアバランチを切り出し、サイズ分布を log-log で描く。

    べき乗フィットの打ち切りは**系のサイズ N**。指標側と同じ値を config から直に読む
    (`analysis/metrics.py` の `resolve_avalanche_smax`)。
    """
    smax = int(window.config.simulation.N)
    sizes = split_avalanches(np.asarray(window.spikes().times, dtype=np.float64)).sizes
    fit = fit_distribution_curves(sizes, fit_max=smax)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_discrete_distribution(ax, fit, xlabel="Avalanche size",
                               regression=delta_cr_fit(sizes, smax=smax))
    if fit.support.size == 0:
        # データが無くても log 軸の枠だけは描いておく。
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(f"Avalanche distribution {window.label}")
    ax.set_xlim(*PAPER_XLIM)
    ax.set_ylim(*PAPER_YLIM)
    save(fig, out_path, dpi=DPI)
