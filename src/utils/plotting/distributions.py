"""離散分布 (アバランシェサイズ・寿命など) の log-log 描画。

フィットそのものは `src.utils.analysis.powerlaw.fit_distribution_curves` が済ませており、
ここは経験 PMF の散布と理論曲線の重ね描き、および凡例の体裁だけを担う。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.analysis.powerlaw import DistributionFit, fit_distribution_curves


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
