"""アバランシェの切り出し。

切り出し方は 2 系統ある。用途に応じて選ぶこと。

- `split_avalanches`      : 平均 ISI を閾値にしたギャップ分割 (Ikeda-Akita-Takahashi 2023)
- `detect_avalanches_binned`: 幅 Δt のビンに離散化し、空ビンで区切られた非空ビンの
                              極大連続列を 1 アバランシェとする (Beggs & Plenz 2003)

分布のフィットは `src.utils.analysis.powerlaw`、臨界性指標は
`src.utils.analysis.criticality` にある。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ======================================================================================
# ギャップ分割 (平均 ISI を閾値にする)
# ======================================================================================

@dataclass
class AvalancheResult:
    sizes: np.ndarray
    starts: np.ndarray
    threshold_ms: float


def split_avalanches(spike_times: np.ndarray) -> AvalancheResult:
    """平均スパイク間隔を閾値にしてアバランチへ分割する。"""
    times = np.sort(np.asarray(spike_times, dtype=np.float64))
    if times.size == 0:
        return AvalancheResult(np.array([], dtype=np.int32), np.array([], dtype=np.float64), np.nan)
    if times.size == 1:
        return AvalancheResult(np.array([1], dtype=np.int32), times.copy(), np.inf)

    intervals = np.diff(times)
    threshold = float(np.mean(intervals))
    split_points = np.where(intervals > threshold)[0] + 1
    groups = np.split(times, split_points)
    sizes = np.array([len(group) for group in groups if len(group) > 0], dtype=np.int32)
    starts = np.array([group[0] for group in groups if len(group) > 0], dtype=np.float64)
    return AvalancheResult(sizes=sizes, starts=starts, threshold_ms=threshold)


# ======================================================================================
# 時間ビン連結 (Beggs & Plenz 2003)
# ======================================================================================

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
