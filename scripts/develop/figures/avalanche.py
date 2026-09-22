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
from scripts.develop.figures import save

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

    べき乗フィットの打ち切りは**系のサイズ N**。指標側と同じ値を config から直に読む
    (`analysis/metrics.py` の `resolve_avalanche_smax`)。
    """
    smax = int(window.config.simulation.N)
    sizes = split_avalanches(np.asarray(window.spikes().times, dtype=np.float64)).sizes
    fit = fit_distribution_curves(sizes, fit_max=smax)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_discrete_distribution(ax, fit, xlabel="Avalanche size")
    if fit.support.size == 0:
        # データが無くても log 軸の枠だけは描いておく。
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(f"Avalanche distribution {window.hour:g} h")
    ax.set_xlim(*PAPER_XLIM)
    ax.set_ylim(*PAPER_YLIM)
    save(fig, out_path, dpi=DPI)
