"""src/utils/beggs_plenz.py の解析関数を合成データで検算する。

実シミュレーション結果には正解が無いため、指数や σ が既知の人工データを与えて
推定量が正しく復元できることを確認する。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import beggs_plenz as bp  # noqa: E402


# --------------------------------------------------------------------------------------
# アバランシェ検出
# --------------------------------------------------------------------------------------

def test_detect_avalanches_on_handmade_bins():
    """空ビンで区切られた非空ビンの連続列が 1 アバランシェになること。

    bin_ms=1.0 で、ビン列が [2, 3, 0, 1, 0, 0, 4, 1] になるようスパイクを置く。
    期待: アバランシェは 3 個 -> サイズ [5, 1, 5]、寿命 [2, 1, 2] ビン。
    """
    times = []
    for bin_index, count in enumerate([2, 3, 0, 1, 0, 0, 4, 1]):
        times.extend([bin_index + 0.5] * count)
    result = bp.detect_avalanches_binned(np.array(times), duration_ms=8.0, bin_ms=1.0)

    assert result.num_avalanches == 3
    assert list(result.sizes) == [5, 1, 5]
    assert list(result.lifetimes_bins) == [2, 1, 2]
    assert list(result.lifetimes_ms) == [2.0, 1.0, 2.0]
    assert list(result.starts_ms) == [0.0, 3.0, 6.0]
    assert list(result.first_bin) == [2, 1, 4]
    # 寿命1ビンのアバランシェは descendants=0 として扱う
    assert list(result.second_bin) == [3, 0, 1]


def test_detect_avalanches_handles_empty_input():
    result = bp.detect_avalanches_binned(np.array([]), duration_ms=100.0, bin_ms=1.0)
    assert result.num_avalanches == 0
    assert result.sizes.size == 0


def test_bin_width_is_floored_at_simulation_dt():
    """平均 IEI が dt を下回っても、ビン幅が dt 未満に潰れないこと。

    シミュレーションのスパイクは dt の整数倍にしか立たないため、dt より細かいビンでは
    非空ビンが必ず空ビンで挟まれ、全アバランシェが寿命1ビンになってしまう
    (σ=0、寿命分布が1点で傾きが nan)。実際に発火率が高い記録窓で起きた不具合。
    """
    dt = 0.1
    rng = np.random.default_rng(4)
    # dt 格子上に多数のスパイクを置く -> 平均 IEI << dt
    steps = rng.integers(0, 1000, size=20_000)
    times = np.sort(steps.astype(np.float64) * dt)
    assert bp.mean_iei_ms(times) < dt  # 前提: IEI が dt を下回っている

    floored = bp.detect_avalanches_binned(times, duration_ms=100.0, bin_ms=None, min_bin_ms=dt)
    assert floored.bin_ms == pytest.approx(dt)
    # 寿命が 1 ビンに潰れていない = 分岐パラメータが意味を持つ
    assert floored.lifetimes_bins.max() > 1
    assert bp.branching_parameter(floored)["sigma_bp"] > 0.0

    # 下限を指定しないと潰れることも確認しておく (退行検知の対照)
    collapsed = bp.detect_avalanches_binned(times, duration_ms=100.0, bin_ms=None)
    assert collapsed.bin_ms < dt
    assert collapsed.lifetimes_bins.max() == 1


def test_default_bin_width_is_mean_iei():
    times = np.array([0.0, 2.0, 4.0, 6.0])  # IEI = 2.0
    assert bp.mean_iei_ms(times) == pytest.approx(2.0)
    result = bp.detect_avalanches_binned(times, duration_ms=8.0, bin_ms=None)
    assert result.bin_ms == pytest.approx(2.0)


# --------------------------------------------------------------------------------------
# べき乗指数の復元
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("true_alpha", [1.5, 2.0, 2.5])
def test_fit_exponent_recovers_known_alpha(true_alpha):
    """既知の指数を持つ離散打ち切りべき乗分布から alpha_mle を復元できること。"""
    rng = np.random.default_rng(12345)
    xmax = 100
    support = np.arange(1, xmax + 1, dtype=np.float64)
    pmf = np.power(support, -true_alpha)
    pmf /= pmf.sum()
    samples = rng.choice(support, size=200_000, p=pmf)

    fit = bp.fit_exponent(samples, xmin=1, xmax=xmax)
    assert fit["alpha_mle"] == pytest.approx(true_alpha, abs=0.05)
    # log-log 回帰の傾きは -alpha 付近 (裾のサンプリングノイズで MLE より粗い)
    assert fit["slope_loglog"] == pytest.approx(-true_alpha, abs=0.25)
    # べき乗分布なので power-law が指数分布より優位
    assert fit["llr"] > 0


def test_fit_exponent_prefers_exponential_for_exponential_data():
    """指数分布データでは LLR が負 (= power-law が不利) になること。"""
    rng = np.random.default_rng(7)
    xmax = 100
    support = np.arange(1, xmax + 1, dtype=np.float64)
    pmf = np.exp(-0.3 * support)
    pmf /= pmf.sum()
    samples = rng.choice(support, size=50_000, p=pmf)

    assert bp.fit_exponent(samples, xmin=1, xmax=xmax)["llr"] < 0


def test_fit_exponent_handles_insufficient_data():
    fit = bp.fit_exponent(np.array([5.0]), xmin=1, xmax=100)
    assert np.isnan(fit["alpha_mle"])


# --------------------------------------------------------------------------------------
# 分岐パラメータ
# --------------------------------------------------------------------------------------

def _simulate_branching_process(sigma, num_avalanches, rng,
                                max_generations=200, max_population=10_000):
    """各世代の子孫数が Poisson(sigma * n) の分岐過程からビン列を作る。

    超臨界 (sigma > 1) では個体数が発散して打ち切らないとメモリを食い潰すため、
    世代数と 1 世代あたりの個体数の両方に上限を設ける。上限に当たると σ の推定値は
    下振れするので、σ の厳密な復元は sigma <= 1 でのみ検証する。
    """
    counts = []
    for _ in range(num_avalanches):
        n = 1
        generation = 0
        while n > 0 and generation < max_generations:
            counts.append(n)
            n = min(int(rng.poisson(sigma * n)), max_population)
            generation += 1
        counts.append(0)  # アバランシェの区切り (空ビン)
    return np.array(counts, dtype=np.int64)


@pytest.mark.parametrize("true_sigma", [0.7, 0.9, 1.0])
def test_branching_parameter_recovers_known_sigma(true_sigma):
    """劣臨界〜臨界の分岐過程から σ を復元できること。

    ビン列を直接渡すのでスパイク時刻への展開は不要 (メモリを食わない)。
    """
    rng = np.random.default_rng(2024)
    counts = _simulate_branching_process(true_sigma, num_avalanches=20_000, rng=rng)
    avalanches = bp.avalanches_from_bin_counts(counts, bin_ms=1.0)

    stats = bp.branching_parameter(avalanches)
    assert stats["sigma_bp"] == pytest.approx(true_sigma, abs=0.05)


def test_branching_parameter_flags_supercritical():
    """超臨界の分岐過程では σ > 1 と判定されること (打ち切りがあるため下限のみ検証)。"""
    rng = np.random.default_rng(11)
    counts = _simulate_branching_process(1.3, num_avalanches=300, rng=rng,
                                         max_generations=40, max_population=2000)
    stats = bp.branching_parameter(bp.avalanches_from_bin_counts(counts, bin_ms=1.0))
    assert stats["sigma_bp"] > 1.05


def test_critical_branching_process_gives_expected_size_exponent():
    """臨界分岐過程 (σ=1) のサイズ分布が α≈1.5 になること (理論値)。"""
    rng = np.random.default_rng(99)
    counts = _simulate_branching_process(1.0, num_avalanches=60_000, rng=rng)
    avalanches = bp.avalanches_from_bin_counts(counts, bin_ms=1.0)

    fit = bp.fit_exponent(avalanches.sizes, xmin=1, xmax=100)
    assert fit["alpha_mle"] == pytest.approx(1.5, abs=0.15)


# --------------------------------------------------------------------------------------
# 相互相関
# --------------------------------------------------------------------------------------

def test_cross_correlogram_reports_undetermined_for_independent_neurons():
    """独立なポアソン発火では相関が検出できず、収束時間は nan (判定不能) になること。

    ここで 0 ms を返してはいけない。相関がそもそも無いことを「即座に収束した」と
    報告すると、判定が偽の合格になる (発火が疎な記録窓で実際に起きた)。
    """
    rng = np.random.default_rng(5)
    duration = 60_000.0
    num_neurons = 40
    times, ids = [], []
    for nid in range(num_neurons):
        n_spikes = rng.poisson(600)
        times.append(rng.uniform(0, duration, n_spikes))
        ids.append(np.full(n_spikes, nid))
    times = np.concatenate(times)
    ids = np.concatenate(ids)

    lags, corr, used = bp.pair_cross_correlogram(
        times, ids, duration, num_pairs=30, bin_ms=5.0, max_lag_ms=500.0,
        rng=np.random.default_rng(1),
    )
    assert used == 30
    assert lags.size == corr.size and lags.size > 0
    assert lags[0] == pytest.approx(-500.0)
    assert np.isnan(bp.correlation_decay_ms(lags, corr))


def test_cross_correlogram_measures_decay_for_correlated_neurons():
    """共通の駆動を受けるニューロン群では、ゼロラグにピークが立ち有限の収束時間が出ること。"""
    rng = np.random.default_rng(17)
    duration = 60_000.0
    num_neurons = 30
    jitter_ms = 15.0
    # 共通イベント時刻の周りに各ニューロンがジッタを持って発火する
    events = rng.uniform(0, duration, 1500)
    times, ids = [], []
    for nid in range(num_neurons):
        participates = events[rng.random(events.size) < 0.6]
        spikes = participates + rng.normal(0.0, jitter_ms, participates.size)
        spikes = spikes[(spikes >= 0) & (spikes < duration)]
        times.append(spikes)
        ids.append(np.full(spikes.size, nid))
    times = np.concatenate(times)
    ids = np.concatenate(ids)

    lags, corr, _ = bp.pair_cross_correlogram(
        times, ids, duration, num_pairs=40, bin_ms=5.0, max_lag_ms=500.0,
        rng=np.random.default_rng(2),
    )
    zero_index = int(np.argmin(np.abs(lags)))
    assert corr[zero_index] == max(corr), "ゼロラグにピークが立っていない"

    decay = bp.correlation_decay_ms(lags, corr)
    assert np.isfinite(decay), "相関があるのに収束時間が判定不能になっている"
    # ジッタ 15 ms なので相関は 100 ms 程度までに消えるはず
    assert 0.0 < decay <= 200.0


def test_cross_correlogram_handles_empty_input():
    lags, corr, used = bp.pair_cross_correlogram(
        np.array([]), np.array([]), 1000.0,
    )
    assert used == 0 and lags.size == 0 and corr.size == 0


def test_correlation_decay_reports_inf_when_never_converging():
    """最終ラグまでノイズ帯へ入らなければ inf を返すこと。"""
    lags = np.arange(-100, 101, 5, dtype=np.float64)
    corr = np.ones_like(lags)  # 常に一定 -> tail の std=0 -> band=0 -> 収束しない
    assert bp.correlation_decay_ms(lags, corr) == float("inf")


# --------------------------------------------------------------------------------------
# 統合 + 判定 + 描画
# --------------------------------------------------------------------------------------

def test_analyze_and_plots_run_end_to_end(tmp_path):
    """臨界分岐過程のスパイク列で解析一式と 3 種類の図が生成できること。"""
    rng = np.random.default_rng(31)
    counts = _simulate_branching_process(1.0, num_avalanches=8000, rng=rng,
                                         max_generations=100, max_population=500)
    # ビン列をスパイク時刻へ展開する (ここは実データと同じ経路を通したいので展開する)。
    total = int(counts.sum())
    bin_of_spike = np.repeat(np.arange(counts.size), counts)
    times = bin_of_spike.astype(np.float64) + rng.random(total)
    ids = rng.integers(0, 50, size=total)

    avalanches, lags, corr, metrics = bp.analyze_avalanches(
        times, ids, duration_ms=float(len(counts)), bin_ms=1.0,
        num_pairs=10, corr_bin_ms=1.0, corr_max_lag_ms=50.0,
        rng=np.random.default_rng(3),
    )

    assert avalanches.num_avalanches > 100
    assert metrics["bin_ms"] == 1.0
    assert metrics["alpha_size"] == pytest.approx(1.5, abs=0.2)
    assert metrics["sigma_bp"] == pytest.approx(1.0, abs=0.1)
    # 判定キーが揃っていること
    for key in ("check_slope_size", "check_slope_lifetime", "check_sigma", "check_corr_decay"):
        assert key in metrics

    bp.plot_size_and_lifetime(avalanches, tmp_path / "dist.png", "test")
    bp.plot_branching(avalanches, tmp_path / "branch.png", "test")
    bp.plot_cross_correlation(lags, corr, tmp_path / "corr.png", "test")
    for name in ("dist.png", "branch.png", "corr.png"):
        assert (tmp_path / name).stat().st_size > 0
