"""Beggs & Plenz (2003, J Neurosci 23(35):11167-11177) 準拠の神経アバランシェ解析。

既存の `src/utils/akita_soc.py` の `split_avalanches` は「平均 ISI を閾値にしたギャップ分割」で
アバランシェを切り出すが、本モジュールは原著の**時間ビン連結**方式を実装する:

    1. スパイク列を幅 Δt のビンに離散化する (Δt の既定値 = 全スパイクの平均イベント間隔 IEI)
    2. 空ビンで区切られた「非空ビンの極大連続列」を 1 アバランシェとする
    3. サイズ = その区間の総スパイク数、寿命 = 連続ビン数 (× Δt で ms)

原著が確認した 4 つの臨界性の指標を計算する:
    - サイズ分布のべき指数     α ≈ -3/2
    - 寿命分布のべき指数       α ≈ -2
    - 分岐パラメータ           σ ≈ 1.0 ± 0.2
    - ペア間クロス相関         100〜200 ms 以内にゼロへ収束

べき乗フィットの最尤推定は `akita_soc` の実装 (Clauset et al. 2009 / Yada et al. 2017 準拠の
離散打ち切り MLE) をそのまま再利用する。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from src.utils.akita_soc import (
    _fit_discrete_exponential_lambda,
    _fit_discrete_powerlaw_alpha,
    log_likelihood_ratio_power_vs_exponential,
)

# 原著が報告する臨界時の参照値と、判定に使う許容幅。
TARGET_SLOPE_SIZE = -1.5
TARGET_SLOPE_LIFETIME = -2.0
TARGET_SIGMA = 1.0
TOLERANCE_SLOPE_SIZE = 0.3
TOLERANCE_SLOPE_LIFETIME = 0.4
TOLERANCE_SIGMA = 0.2
TARGET_CORR_DECAY_MS = 200.0


@dataclass
class BinnedAvalanches:
    """時間ビン連結で切り出したアバランシェ群。全配列はアバランシェ index で整合する。"""
    sizes: np.ndarray           # int64, 各アバランシェの総スパイク数
    lifetimes_bins: np.ndarray  # int64, 連続ビン数
    lifetimes_ms: np.ndarray    # float64
    starts_ms: np.ndarray       # float64, 先頭ビンの開始時刻
    first_bin: np.ndarray       # int64, 先頭ビンのスパイク数 (= ancestors)
    second_bin: np.ndarray      # int64, 2 番目のビンのスパイク数 (= descendants)
    bin_ms: float
    bin_counts: np.ndarray      # int64, 全ビンのスパイク数 (分岐パラメータの別推定に使う)

    @property
    def num_avalanches(self) -> int:
        return int(self.sizes.size)


def _empty_avalanches(bin_ms: float) -> BinnedAvalanches:
    empty_i = np.array([], dtype=np.int64)
    empty_f = np.array([], dtype=np.float64)
    return BinnedAvalanches(
        sizes=empty_i, lifetimes_bins=empty_i, lifetimes_ms=empty_f, starts_ms=empty_f,
        first_bin=empty_i, second_bin=empty_i, bin_ms=bin_ms, bin_counts=empty_i,
    )


def mean_iei_ms(spike_times: np.ndarray) -> float:
    """全スパイクをひとつのイベント列とみなした平均イベント間隔 [ms]。

    Beggs & Plenz が採用する既定のビン幅 Δt。スパイクが 2 個未満なら nan。
    """
    times = np.sort(np.asarray(spike_times, dtype=np.float64))
    if times.size < 2:
        return float("nan")
    return float(np.mean(np.diff(times)))


def detect_avalanches_binned(
    spike_times: np.ndarray,
    duration_ms: float,
    bin_ms: float | None = None,
    min_bin_ms: float | None = None,
) -> BinnedAvalanches:
    """時間ビン連結方式でアバランシェを検出する。

    Args:
        spike_times: スパイク時刻 [ms] (ソート不要)
        duration_ms: 解析対象の総時間 [ms]
        bin_ms: ビン幅 [ms]。None なら `mean_iei_ms(spike_times)` を使う。
        min_bin_ms: ビン幅の下限 [ms]。**シミュレーションの dt を渡すこと。**

    空ビンで挟まれた非空ビンの極大連続列が 1 アバランシェ。

    min_bin_ms が必要な理由: シミュレーションではスパイクが dt の整数倍の時刻にしか
    立たないため、dt より細かいビンを使うと非空ビンが必ず空ビンで挟まれ、すべての
    アバランシェが「寿命1ビン」に潰れてしまう (σ=0、寿命分布が1点)。発火率が高くて
    平均 IEI が dt を下回るときに実際に起きるので、下限で丸める。
    """
    times = np.asarray(spike_times, dtype=np.float64)
    if bin_ms is None:
        bin_ms = mean_iei_ms(times)
    if min_bin_ms is not None and np.isfinite(bin_ms) and np.isfinite(min_bin_ms):
        bin_ms = max(float(bin_ms), float(min_bin_ms))
    if not np.isfinite(bin_ms) or bin_ms <= 0.0 or duration_ms <= 0.0 or times.size == 0:
        return _empty_avalanches(float(bin_ms) if np.isfinite(bin_ms) else float("nan"))

    num_bins = max(1, int(np.ceil(duration_ms / bin_ms)))
    edges = np.arange(num_bins + 1, dtype=np.float64) * bin_ms
    counts = np.histogram(times, bins=edges)[0].astype(np.int64)
    return avalanches_from_bin_counts(counts, bin_ms)


def avalanches_from_bin_counts(bin_counts: np.ndarray, bin_ms: float) -> BinnedAvalanches:
    """既にビン化済みのスパイク数列からアバランシェを切り出す。

    `detect_avalanches_binned` の実体。ビン列を直接持っている場合 (合成データの検証など)
    はスパイク時刻へ展開せずにこちらを使う。
    """
    counts = np.asarray(bin_counts, dtype=np.int64)
    if counts.size == 0:
        return _empty_avalanches(float(bin_ms))

    occupied = counts > 0
    if not occupied.any():
        return _empty_avalanches(float(bin_ms))

    # 非空ビンの極大連続列の境界を立ち上がり/立ち下がりで求める。
    padded = np.concatenate(([False], occupied, [False]))
    edges_diff = np.diff(padded.astype(np.int8))
    starts = np.nonzero(edges_diff == 1)[0]
    stops = np.nonzero(edges_diff == -1)[0]  # 排他的な終端

    # 区間和は累積和の差で一括計算する (アバランシェ数が多くてもループしない)。
    cumulative = np.concatenate(([0], np.cumsum(counts)))
    sizes = (cumulative[stops] - cumulative[starts]).astype(np.int64)
    lifetimes_bins = (stops - starts).astype(np.int64)

    first_bin = counts[starts]
    second_index = starts + 1
    has_second = lifetimes_bins >= 2
    second_bin = np.zeros_like(first_bin)
    second_bin[has_second] = counts[second_index[has_second]]

    return BinnedAvalanches(
        sizes=sizes,
        lifetimes_bins=lifetimes_bins,
        lifetimes_ms=lifetimes_bins.astype(np.float64) * bin_ms,
        starts_ms=starts.astype(np.float64) * bin_ms,
        first_bin=first_bin.astype(np.int64),
        second_bin=second_bin.astype(np.int64),
        bin_ms=float(bin_ms),
        bin_counts=counts,
    )


def branching_parameter(avalanches: BinnedAvalanches) -> dict[str, float]:
    """分岐パラメータ σ を 2 通りの推定量で計算する。

    sigma_bp:
        Beggs & Plenz の定義。各アバランシェについて「先頭ビンのスパイク数 (ancestors) に
        対する 2 番目のビンのスパイク数 (descendants) の比」を取り、全アバランシェで平均する。
        寿命 1 ビンのアバランシェは descendants=0 として算入する (これを除くと σ が
        系統的に過大評価される)。
    sigma_bins:
        参考値。連続する全ビン対 (n_t > 0) について n_{t+1}/n_t を平均したもの。
        アバランシェ境界をまたがないぶん sigma_bp より滑らかだが原著の定義ではない。

    σ ≈ 1 が臨界、< 1 が劣臨界 (活動が消える)、> 1 が超臨界 (活動が爆発する)。
    """
    result = {"sigma_bp": float("nan"), "sigma_bins": float("nan"),
              "num_avalanches": float(avalanches.num_avalanches)}

    valid = avalanches.first_bin > 0
    if valid.any():
        ratios = avalanches.second_bin[valid] / avalanches.first_bin[valid]
        result["sigma_bp"] = float(np.mean(ratios))

    counts = avalanches.bin_counts
    if counts.size >= 2:
        current = counts[:-1]
        nxt = counts[1:]
        nonzero = current > 0
        if nonzero.any():
            result["sigma_bins"] = float(np.mean(nxt[nonzero] / current[nonzero]))

    return result


def discrete_distribution(values: np.ndarray, xmax: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """正の整数値の経験 PMF を (support, prob) で返す (値が現れた点のみ)。"""
    data = np.asarray(values, dtype=np.int64)
    data = data[data >= 1]
    if xmax is not None:
        data = data[data <= xmax]
    if data.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    counts = np.bincount(data)[1:]
    support = np.arange(1, counts.size + 1)
    mask = counts > 0
    return support[mask], counts[mask] / data.size


def fit_exponent(values: np.ndarray, xmin: int = 1, xmax: int = 100) -> dict[str, float]:
    """べき乗指数を最尤推定 + log-log 回帰の両方で求める。

    Returns:
        alpha_mle   : 離散打ち切り MLE の指数 (正値。p(x) ∝ x^-alpha)
        slope_loglog: log-log 平面での最小二乗回帰の傾き (負値。論文の α に対応)
        llr         : power-law vs exponential の対数尤度比 (正 → power-law 優位)
        num_samples : フィットに使ったサンプル数
    """
    data = np.asarray(values, dtype=np.float64)
    data = data[(data >= xmin) & (data <= xmax)]
    out = {
        "alpha_mle": float("nan"),
        "slope_loglog": float("nan"),
        "llr": float("nan"),
        "num_samples": float(data.size),
    }
    if data.size < 2:
        return out

    out["alpha_mle"] = _fit_discrete_powerlaw_alpha(data, xmin, xmax)
    out["llr"] = log_likelihood_ratio_power_vs_exponential(data, smax=xmax)

    support, prob = discrete_distribution(data.astype(np.int64), xmax=xmax)
    if support.size >= 2:
        slope, _ = np.polyfit(np.log(support), np.log(prob), 1)
        out["slope_loglog"] = float(slope)
    return out


def pair_cross_correlogram(
    spike_times: np.ndarray,
    spike_ids: np.ndarray,
    duration_ms: float,
    num_pairs: int = 200,
    bin_ms: float = 5.0,
    max_lag_ms: float = 500.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """ランダムなニューロンペアの正規化相互相関をプールして平均する。

    単一ペアのスパイク列はスパースすぎて相関がノイズに埋もれるため、多数のペアの
    相関関数を平均してから収束時間を測る。

    Returns:
        (lags_ms, mean_corr, num_pairs_used)
    """
    times = np.asarray(spike_times, dtype=np.float64)
    ids = np.asarray(spike_ids, dtype=np.int64)
    empty = np.array([], dtype=np.float64)
    if times.size == 0 or duration_ms <= 0.0 or bin_ms <= 0.0:
        return empty, empty, 0

    if rng is None:
        rng = np.random.default_rng(0)

    # 十分にスパイクしているニューロンだけを候補にする (相関が定義できないため)
    unique_ids, counts = np.unique(ids, return_counts=True)
    candidates = unique_ids[counts >= 2]
    if candidates.size < 2:
        return empty, empty, 0

    num_bins = max(2, int(np.ceil(duration_ms / bin_ms)))
    max_lag_bins = max(1, int(round(max_lag_ms / bin_ms)))
    max_lag_bins = min(max_lag_bins, num_bins - 1)

    # 必要なニューロンのビン列だけを作る (全ニューロン分は作らない)
    selected = rng.choice(candidates, size=min(num_pairs * 2, candidates.size), replace=False)
    position_of = {int(nid): pos for pos, nid in enumerate(selected)}
    trains = np.zeros((selected.size, num_bins), dtype=np.float64)
    keep = np.isin(ids, selected)
    bin_index = np.minimum((times[keep] / bin_ms).astype(np.int64), num_bins - 1)
    row_index = np.array([position_of[int(i)] for i in ids[keep]], dtype=np.int64)
    np.add.at(trains, (row_index, bin_index), 1.0)

    # 平均を引き、標準偏差で正規化しておくと相関が Pearson 係数になる
    trains -= trains.mean(axis=1, keepdims=True)
    stds = trains.std(axis=1)
    usable = np.nonzero(stds > 0)[0]
    if usable.size < 2:
        return empty, empty, 0

    accumulator = np.zeros(2 * max_lag_bins + 1, dtype=np.float64)
    used = 0
    for _ in range(num_pairs):
        a, b = rng.choice(usable, size=2, replace=False)
        full = signal.correlate(trains[a], trains[b], mode="full", method="fft")
        centre = num_bins - 1
        window = full[centre - max_lag_bins: centre + max_lag_bins + 1]
        accumulator += window / (stds[a] * stds[b] * num_bins)
        used += 1

    lags_ms = np.arange(-max_lag_bins, max_lag_bins + 1, dtype=np.float64) * bin_ms
    return lags_ms, accumulator / max(used, 1), used


def correlation_decay_ms(
    lags_ms: np.ndarray,
    corr: np.ndarray,
    noise_fraction: float = 0.5,
    outlier_tolerance: float = 0.05,
) -> float:
    """相関が「ノイズ帯に収まったまま戻らない」最小の正ラグ [ms] を返す。

    ノイズ帯は、ラグ窓の外側 `noise_fraction` 側 (絶対ラグが大きい領域) の相関の標準偏差の
    2 倍とする。

    判定は「そのラグ以降でノイズ帯を外れる点の割合が `outlier_tolerance` 以下」。
    単に「最後に帯を外れた点の次」とすると、純粋なノイズでも 2σ 超えが偶然 5% 程度
    出るため、大ラグ側の 1 点のはずれで収束時刻が窓の端まで飛んでしまう。割合で見ることで
    「実質的にノイズと区別できなくなる点」を安定に取れる。

    最終ラグまで条件を満たさなければ inf、判定不能なら nan。

    重要: ゼロラグのピーク自体がノイズ帯に埋もれている場合は **nan** を返す。
    発火が疎すぎてペアの相関がそもそも検出できないとき、素直に「ラグ0で既にノイズ内」=
    収束時間0 と報告すると、相関が無いことが「即座に収束した」という偽の合格になる。
    相関の有無と減衰の速さは別の主張なので、検出できないときは判定不能とする。
    """
    lags = np.asarray(lags_ms, dtype=np.float64)
    values = np.asarray(corr, dtype=np.float64)
    if lags.size == 0 or lags.size != values.size:
        return float("nan")

    threshold_lag = np.max(np.abs(lags)) * noise_fraction
    tail = values[np.abs(lags) >= threshold_lag]
    if tail.size < 2:
        return float("nan")
    band = 2.0 * float(np.std(tail))

    positive = lags >= 0
    pos_lags = lags[positive]
    if pos_lags.size == 0:
        return float("nan")

    if not np.any(values):
        return float("nan")  # 相関がまったく計算できていない

    # ゼロラグにそもそも有意なピークが無ければ「相関が検出できない」= 判定不能。
    # (band<=0 の縮退時はこの判定を飛ばし、下の suffix 判定で inf になる)
    zero_index = int(np.argmin(np.abs(lags)))
    if band > 0.0 and abs(values[zero_index]) <= band:
        return float("nan")

    outside = (np.abs(values[positive]) > band).astype(np.float64)

    # 各開始位置 i について、[i, 末尾] で帯を外れる点の割合を一括計算する。
    outside_suffix = np.cumsum(outside[::-1])[::-1]
    remaining = np.arange(pos_lags.size, 0, -1, dtype=np.float64)
    fraction_outside = outside_suffix / remaining

    qualifying = np.nonzero(fraction_outside <= outlier_tolerance)[0]
    if qualifying.size == 0:
        return float("inf")
    return float(pos_lags[int(qualifying[0])])


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

def _plot_distribution_panel(ax, values, xmax, target_slope, xlabel):
    support, prob = discrete_distribution(values, xmax=None)
    if support.size == 0:
        ax.set_xlabel(xlabel)
        return
    ax.scatter(support, prob, s=12, color="black", label="empirical", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    fit = fit_exponent(values, xmin=1, xmax=xmax)
    data = np.asarray(values, dtype=np.float64)
    data = data[(data >= 1) & (data <= xmax)]
    if data.size >= 2:
        fit_support = np.arange(1, xmax + 1, dtype=np.float64)
        emp_mass = float(prob[support <= xmax].sum())

        p_pow = np.power(fit_support, -fit["alpha_mle"])
        p_pow = p_pow / p_pow.sum() * emp_mass
        ax.plot(fit_support, p_pow, color="tab:red", lw=1.5, zorder=2,
                label=f"power-law MLE (α={fit['alpha_mle']:.2f})")

        lam = _fit_discrete_exponential_lambda(data, 1, xmax)
        p_exp = np.exp(-lam * fit_support)
        p_exp = p_exp / p_exp.sum() * emp_mass
        ax.plot(fit_support, p_exp, color="tab:blue", lw=1.2, ls=":", zorder=2,
                label=f"exponential MLE (λ={lam:.3f})")

        # 原著の参照傾き。経験分布の先頭に合わせて配置する。
        reference = prob[0] * np.power(fit_support, target_slope)
        ax.plot(fit_support, reference, color="gray", lw=1.2, ls="--", zorder=1,
                label=f"reference slope {target_slope}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    ax.legend(fontsize=6.5, loc="lower left",
              title=f"fit slope={fit['slope_loglog']:.2f}  LLR={fit['llr']:.0f}",
              title_fontsize=6.5)


def plot_size_and_lifetime(
    avalanches: BinnedAvalanches,
    out_path: Path,
    title: str,
    size_fit_max: int = 100,
    lifetime_fit_max: int = 50,
) -> None:
    """アバランシェのサイズ分布と寿命分布を 2 パネルの log-log で描く。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    _plot_distribution_panel(axes[0], avalanches.sizes, size_fit_max,
                             TARGET_SLOPE_SIZE, "Avalanche size [spikes]")
    axes[0].set_title("Size distribution")

    _plot_distribution_panel(axes[1], avalanches.lifetimes_bins, lifetime_fit_max,
                             TARGET_SLOPE_LIFETIME, f"Lifetime [bins of {avalanches.bin_ms:.2f} ms]")
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
