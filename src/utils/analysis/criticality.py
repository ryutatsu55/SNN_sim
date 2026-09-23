"""臨界性の指標。

アバランシェサイズ分布や発火列から、系が臨界・劣臨界・超臨界のどこにいるかを
測る量を計算する。Beggs & Plenz (2003) の合否判定 (許容幅つきのチェック) も、しきい値は
判定であって図ではないので末尾に置いてある。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class DeltaCrFit:
    """ΔCr の中身。図が回帰直線を重ね描きするためにここから取る。

    `prob` / `fit` は `grid` (= 1..smax) 上で揃っている。観測されなかったサイズの
    `prob` は 0 (回帰には使っていない)。
    """
    grid: np.ndarray        # 1..smax
    prob: np.ndarray        # pemp(s)。観測が無ければ 0
    fit: np.ndarray         # pfit(s) = 10^intercept · s^slope (再正規化しない)
    slope: float            # log-log 平面での傾き (負値)
    intercept: float
    smin: int
    upper: float            # Aupper (S22)
    lower: float            # Alower (S23)
    delta_cr: float         # ΔCr (S24)


def delta_cr_fit(
    sizes: np.ndarray, smax: int = 100, smin: int = 1, min_points: int = 10
) -> DeltaCrFit | None:
    """ΔCr とその回帰直線を返す。フィットできなければ `None`。

    **`criticality_index_delta_cr()` の実体。** 図が同じ直線を描けるように、
    スカラーだけでなく直線そのものを返す口を分けてある (2 通りに実装しない)。
    定義と引数は `criticality_index_delta_cr()` の docstring を読むこと。
    """
    data = np.asarray(sizes, dtype=np.int64)
    data = data[(data >= 1) & (data <= smax)]
    if data.size == 0:
        return None

    grid = np.arange(1, smax + 1)
    counts = np.bincount(data, minlength=smax + 1)[1:smax + 1]
    prob = counts / data.size
    # 回帰は観測点だけで引く (log10(0) が引けないため)。和は S21 どおり smin..smax の
    # 全整数サイズで取るので、観測されなかったサイズは pemp=0 として −pfit を寄与する。
    fitted_on = (counts > 0) & (grid >= smin)
    if np.count_nonzero(fitted_on) < min_points:
        return None

    log_grid = np.log10(grid)
    # **観測個数で重み付ける。** polyfit の w は二乗前の残差に掛かるので、
    # w=sqrt(count) が「重み = count」の加重最小二乗になる。
    slope, intercept = np.polyfit(log_grid[fitted_on], np.log10(prob[fitted_on]), 1,
                                  w=np.sqrt(counts[fitted_on]))
    fit = np.power(10.0, intercept + slope * log_grid)

    evaluated_on = grid >= smin
    deviation = prob[evaluated_on] - fit[evaluated_on]
    upper = float(np.sum(np.maximum(deviation, 0.0)))
    lower = float(np.sum(np.minimum(deviation, 0.0)))
    return DeltaCrFit(
        grid=grid, prob=prob, fit=fit, slope=float(slope), intercept=float(intercept),
        smin=int(smin), upper=upper, lower=lower,
        delta_cr=upper if abs(upper) >= abs(lower) else lower,
    )


def criticality_index_delta_cr(
    sizes: np.ndarray, smax: int = 100, smin: int = 1, min_points: int = 10
) -> float:
    """臨界性指標 ΔCr (Ikeda-Akita-Takahashi 2023 supplementary 式 S21-S24)。

    経験分布 pemp(s) と、その log-log プロットへの線形回帰が与えるべき乗 pfit(s) の差を
    上振れ/下振れに分けて足し、絶対値の大きい方を符号付きで返す:

        Aupper = Σ_{s=smin..smax} max(pemp(s) − pfit(s), 0)    … S22
        Alower = Σ_{s=smin..smax} min(pemp(s) − pfit(s), 0)    … S23
        ΔCr    = |Aupper| ≥ |Alower| なら Aupper, そうでなければ Alower   … S24

    **正 = 超臨界、≈0 = 臨界、負 = 劣臨界。**

    差を取るのは**確率そのもの** (S21 の pemp(s) − pfit(s)) であって log ではない。
    log-log なのは回帰を引く平面だけで、pfit はその直線を確率へ戻したもの
    (pfit(s) = 10^intercept · s^slope, 再正規化しない)。
    **回帰と残差を同じ log 空間で取ると最小二乗の性質から Σ残差 = 0 になり、
    Aupper = |Alower| で符号が消える。** 確率空間の差にはこの縮退が無いので、
    smin による範囲の切り分けは符号のために必要ではない。

    Args:
        sizes: アバランシェサイズの標本 (1 以上の整数)。
        smax: 考えるサイズの上限。**アバランシェでは系のニューロン数 N。**
            これを超えるサイズは pemp を作る前に捨てる (論文 II.B と同じ)。
        smin: 考えるサイズの下限。回帰も和もここから始める。既定の 1 は
            **下限を切らない**の意 (サイズは 1 以上なので smin=0 と同義)。
        min_points: 回帰に使える観測点 (pemp > 0 のサイズ) がこれを下回ったら
            `nan` を返す。**点が少ないと直線の傾きが 1〜2 個の裾で決まり、
            ΔCr が指標ではなく乱数になる**ための足切り。

    論文は smin を「線形フィットの二乗誤差和が最小になるよう決めた」と書くが、
    その規準は点を減らすほど誤差が減るため走査範囲の上限に張り付く (再現しない)。
    平均二乗誤差や Clauset の KS 規準に替えても、0h(純ポアソン)では裾へ逃げて
    ΔCr ≈ 0 になり、論文の −0.23 を出さない。**恣意的な較正を避けるため既定では
    切らない。** 別の値を使うときは明示的に渡すこと。

    **回帰は各サイズの観測個数で重み付ける (加重最小二乗)。** 経験 PMF の点は精度が
    そろっていない —— log10(pemp(s)) の分散は 1/count(s) に比例するので、数個しか
    観測が無い裾と数千個ある s=1 を等価に扱うと、直線が裾に引きずられて小サイズ側の
    切片が跳ね上がる (素の最小二乗では pfit(1) が確率 1 を超えることがある)。
    逆分散重み = count がこの分布に対する正しい回帰で、副作用として値の尺度が
    標本数によらず安定し、Tetzlaff et al. 2010 の判定幅 ±0.195 と同じ世界に収まる。

    **符号と時間発展で読むこと。** 論文の絶対値とは一致しない
    (実測と経緯は `docs/technical/akita_soc_reproduction_memo.md` §12)。
    """
    fit = delta_cr_fit(sizes, smax=smax, smin=smin, min_points=min_points)
    return np.nan if fit is None else fit.delta_cr


def burstiness_index(spike_times: np.ndarray, duration_ms: float, bin_ms: float = 1000.0) -> float:
    if duration_ms <= 0:
        return np.nan
    bins = np.arange(0.0, duration_ms + bin_ms, bin_ms)
    if bins.size < 2:
        return np.nan
    counts, _ = np.histogram(spike_times, bins=bins)
    total = int(np.sum(counts))
    if total == 0:
        return np.nan
    sorted_counts = np.sort(counts)[::-1]
    top_n = max(1, int(np.ceil(0.15 * sorted_counts.size)))
    return float(((np.sum(sorted_counts[:top_n]) / total) - 0.15) / 0.85)


def bimodality_d(sizes: np.ndarray) -> float:
    """Bimodality 指標 D（Ikeda-Akita-Takahashi 2023 supplementary 式 S26）。

    アバランシェサイズ S1,...,SN を降順ソートし、隣接サイズの最大差（最大ギャップ）を D とする:
        D = max_i (S_i − S_{i+1})   （S_i は降順、gap は非負）
    論文本文の表記 `max_i S_{i+1} − S_i` は符号の綴りで、意図は「avalanche size の
    最大差」＝最大ギャップ（Yada et al. 2017 に準拠）。二峰性（小アバランシェ群と
    系サイズ級バーストの間のギャップ）で D が大きくなる。
    ※ サイズは上限を設けない（power-law fit の smax=100 とは別。S26 は全サイズを使う）。
    """
    sorted_sizes = np.sort(np.asarray(sizes, dtype=np.float64))[::-1]
    if sorted_sizes.size < 2:
        return np.nan
    return float(np.max(sorted_sizes[:-1] - sorted_sizes[1:]))


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


# ======================================================================================
# Beggs & Plenz (2003, J Neurosci 23(35):11167-11177) の合否判定
#
# 原著が確認した 4 つの臨界性の指標と、その許容幅。**数値そのものを測るのは上の関数群**
# で、ここが持つのは「どこまでを合格とするか」という**しきい値だけ**。
#
# ここに置いてある理由: しきい値は図ではなく判定であり、判定は実験をまたいで同じ意味を
# 持つ (損傷実験が「切断後に臨界から外れたか」を問うときも同じ基準を使う)。以前は
# `src/utils/experiments/beggs_plenz.py` にあったが、そのモジュールは論文名を冠している
# だけで実験の実体 (回すスクリプト) を持っていなかった。
# ======================================================================================

TARGET_SLOPE_SIZE = -1.5
TARGET_SLOPE_LIFETIME = -2.0
TARGET_SIGMA = 1.0
TOLERANCE_SLOPE_SIZE = 0.3
TOLERANCE_SLOPE_LIFETIME = 0.4
TOLERANCE_SIGMA = 0.2
TARGET_CORR_DECAY_MS = 200.0


def criticality_checks(metrics: dict[str, float]) -> dict[str, bool]:
    """4 つの臨界性判定を bool で返す。`metrics` は `avalanche.analyze_avalanches` の出力。"""
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
