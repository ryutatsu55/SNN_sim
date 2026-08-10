"""Beggs & Plenz (2003, J Neurosci 23(35):11167-11177) 準拠の神経アバランシェ解析。

原著が確認した 4 つの臨界性の指標を、この論文の許容幅で判定する:
    - サイズ分布のべき指数     α ≈ -3/2
    - 寿命分布のべき指数       α ≈ -2
    - 分岐パラメータ           σ ≈ 1.0 ± 0.2
    - ペア間クロス相関         100〜200 ms 以内にゼロへ収束

アバランシェの切り出し (時間ビン連結)、べき乗フィット、分岐パラメータ、相互相関そのものは
`src/utils/analysis/` にある汎用実装を使う。ここに残すのは**この論文固有のもの**、すなわち
参照値と許容幅・合否判定・図の体裁 (参照傾きを重ねた 2 パネルなど) に限る。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.analysis.avalanche import (
    BinnedAvalanches,
    branching_parameter,
    detect_avalanches_binned,
)
from src.utils.analysis.criticality import correlation_decay_ms, pair_cross_correlogram
from src.utils.analysis.powerlaw import fit_distribution_curves, fit_exponent
from src.utils.plotting.distributions import plot_discrete_distribution

# 原著が報告する臨界時の参照値と、判定に使う許容幅。
TARGET_SLOPE_SIZE = -1.5
TARGET_SLOPE_LIFETIME = -2.0
TARGET_SIGMA = 1.0
TOLERANCE_SLOPE_SIZE = 0.3
TOLERANCE_SLOPE_LIFETIME = 0.4
TOLERANCE_SIGMA = 0.2
TARGET_CORR_DECAY_MS = 200.0


def criticality_checks(metrics: dict[str, float]) -> dict[str, bool]:
    """4 つの臨界性判定を bool で返す。metrics は `analyze_avalanches` の出力。"""
    def near(value, target, tol):
        return bool(np.isfinite(value) and abs(value - target) <= tol)

    decay = metrics.get("corr_decay_ms", float("nan"))
    return {
        "check_slope_size": near(metrics.get("slope_size", np.nan),
                                 TARGET_SLOPE_SIZE, TOLERANCE_SLOPE_SIZE),
        "check_slope_lifetime": near(metrics.get("slope_lifetime", np.nan),
                                     TARGET_SLOPE_LIFETIME, TOLERANCE_SLOPE_LIFETIME),
        "check_sigma": near(metrics.get("sigma_bp", np.nan), TARGET_SIGMA, TOLERANCE_SIGMA),
        "check_corr_decay": bool(np.isfinite(decay) and decay <= TARGET_CORR_DECAY_MS),
    }


def analyze_avalanches(
    spike_times: np.ndarray,
    spike_ids: np.ndarray,
    duration_ms: float,
    bin_ms: float | None = None,
    min_bin_ms: float | None = None,
    size_fit_max: int = 100,
    lifetime_fit_max: int = 50,
    num_pairs: int = 200,
    corr_bin_ms: float = 5.0,
    corr_max_lag_ms: float = 500.0,
    rng: np.random.Generator | None = None,
) -> tuple[BinnedAvalanches, np.ndarray, np.ndarray, dict[str, float]]:
    """アバランシェ検出・分布フィット・分岐パラメータ・相互相関を一括で計算する。

    `min_bin_ms` にはシミュレーションの dt を渡すこと (理由は
    `detect_avalanches_binned` の docstring を参照)。

    Returns:
        (avalanches, lags_ms, mean_corr, metrics)
    """
    avalanches = detect_avalanches_binned(spike_times, duration_ms,
                                          bin_ms=bin_ms, min_bin_ms=min_bin_ms)
    size_fit = fit_exponent(avalanches.sizes, xmin=1, xmax=size_fit_max)
    lifetime_fit = fit_exponent(avalanches.lifetimes_bins, xmin=1, xmax=lifetime_fit_max)
    branching = branching_parameter(avalanches)
    lags, corr, pairs_used = pair_cross_correlogram(
        spike_times, spike_ids, duration_ms,
        num_pairs=num_pairs, bin_ms=corr_bin_ms, max_lag_ms=corr_max_lag_ms, rng=rng,
    )

    metrics: dict[str, float] = {
        "bin_ms": avalanches.bin_ms,
        "num_avalanches": float(avalanches.num_avalanches),
        "alpha_size": size_fit["alpha_mle"],
        "slope_size": size_fit["slope_loglog"],
        "llr_size": size_fit["llr"],
        "alpha_lifetime": lifetime_fit["alpha_mle"],
        "slope_lifetime": lifetime_fit["slope_loglog"],
        "llr_lifetime": lifetime_fit["llr"],
        "sigma_bp": branching["sigma_bp"],
        "sigma_bins": branching["sigma_bins"],
        "corr_decay_ms": correlation_decay_ms(lags, corr),
        "corr_num_pairs": float(pairs_used),
        "max_avalanche_size": float(avalanches.sizes.max()) if avalanches.num_avalanches else float("nan"),
        "max_lifetime_ms": float(avalanches.lifetimes_ms.max()) if avalanches.num_avalanches else float("nan"),
    }
    metrics.update({k: float(v) for k, v in criticality_checks(metrics).items()})
    return avalanches, lags, corr, metrics


# ======================================================================================
# 可視化
# ======================================================================================

def plot_size_and_lifetime(
    avalanches: BinnedAvalanches,
    out_path: Path,
    title: str,
    size_fit_max: int = 100,
    lifetime_fit_max: int = 50,
) -> None:
    """アバランシェのサイズ分布と寿命分布を 2 パネルの log-log で描く。

    原著の参照傾き (サイズ -3/2、寿命 -2) を重ねるのがこの実験固有の見せ方で、
    パネル 1 枚の描画そのものは汎用の `plot_discrete_distribution` に任せる。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    plot_discrete_distribution(
        axes[0], fit_distribution_curves(avalanches.sizes, fit_max=size_fit_max),
        xlabel="Avalanche size [spikes]", reference_slope=TARGET_SLOPE_SIZE,
    )
    axes[0].set_title("Size distribution")

    plot_discrete_distribution(
        axes[1], fit_distribution_curves(avalanches.lifetimes_bins, fit_max=lifetime_fit_max),
        xlabel=f"Lifetime [bins of {avalanches.bin_ms:.2f} ms]",
        reference_slope=TARGET_SLOPE_LIFETIME,
    )
    axes[1].set_title("Lifetime distribution")

    fig.suptitle(f"{title}  (Δt={avalanches.bin_ms:.2f} ms, n={avalanches.num_avalanches})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_branching(avalanches: BinnedAvalanches, out_path: Path, title: str) -> None:
    """descendants(2番目のビン) vs ancestors(先頭ビン) の散布図と σ を描く。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = branching_parameter(avalanches)
    fig, ax = plt.subplots(figsize=(5, 4.5))

    if avalanches.num_avalanches:
        ancestors = avalanches.first_bin
        descendants = avalanches.second_bin
        ax.scatter(ancestors, descendants, s=8, alpha=0.3, color="black",
                   edgecolors="none", label="avalanches")

        # ancestors ごとの平均 descendants (= 分岐比の期待値の直接推定)
        max_a = int(ancestors.max())
        if max_a >= 1:
            totals = np.bincount(ancestors, weights=descendants, minlength=max_a + 1)
            occurrences = np.bincount(ancestors, minlength=max_a + 1)
            valid = occurrences > 0
            grid = np.arange(max_a + 1)[valid]
            means = totals[valid] / occurrences[valid]
            ax.plot(grid, means, color="tab:red", lw=1.5, marker="o", ms=3,
                    label="mean descendants")
            ax.plot(grid, grid, color="gray", ls="--", lw=1.2, label="σ = 1")

    ax.set_xlabel("Ancestors (spikes in first bin)")
    ax.set_ylabel("Descendants (spikes in second bin)")
    ax.set_title(f"{title}\nσ_bp={stats['sigma_bp']:.3f}  (σ_bins={stats['sigma_bins']:.3f})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cross_correlation(
    lags_ms: np.ndarray,
    corr: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """プール済みペア相互相関と、収束時刻・100/200 ms の目安を描く。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))

    lags = np.asarray(lags_ms, dtype=np.float64)
    values = np.asarray(corr, dtype=np.float64)
    if lags.size:
        ax.plot(lags, values, color="black", lw=1.0)
        ax.axhline(0.0, color="gray", lw=0.8)

        threshold_lag = np.max(np.abs(lags)) * 0.5
        tail = values[np.abs(lags) >= threshold_lag]
        if tail.size >= 2:
            band = 2.0 * float(np.std(tail))
            ax.axhspan(-band, band, color="tab:orange", alpha=0.15,
                       label="noise band (±2σ of tail)")

        decay = correlation_decay_ms(lags, values)
        if np.isfinite(decay):
            ax.axvline(decay, color="tab:red", lw=1.3, ls="-",
                       label=f"decay to noise: {decay:.0f} ms")
        else:
            ax.plot([], [], " ", label="no convergence within window")

        for reference in (100.0, 200.0):
            ax.axvline(reference, color="tab:blue", lw=0.9, ls="--", alpha=0.7)
        ax.plot([], [], color="tab:blue", ls="--", lw=0.9, label="100 / 200 ms")

    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel("Normalized cross-correlation")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
