"""ΔCr (Ikeda-Akita-Takahashi 2023 S21-S24 / Tetzlaff 2010 の Δp) の性質を固定する。

ΔCr は「log-log プロット上での経験点と回帰直線の縦方向のずれの平均」であり、
Tetzlaff が sub/critical/super の境目に置いた |Δp| = 0.195 と同じスケールに乗る。
ここで縛るのは (a) 3 つの状態に対する符号、(b) 値のスケール、(c) 純べき乗で ~0。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.analysis.criticality import criticality_index_delta_cr

SMAX = 100
THRESHOLD = 0.195  # Tetzlaff et al. 2010 の判定閾値


def _sample(pmf: np.ndarray, n: int, seed: int) -> np.ndarray:
    grid = np.arange(1, SMAX + 1)
    return np.random.default_rng(seed).choice(grid, size=n, p=pmf / pmf.sum())


def _powerlaw(alpha: float) -> np.ndarray:
    return np.arange(1, SMAX + 1, dtype=float) ** -alpha


@pytest.mark.parametrize("alpha", [1.5, 1.7, 2.0])
def test_pure_power_law_is_critical(alpha):
    """べき乗そのものは臨界: |ΔCr| が判定閾値の内側に入る。"""
    value = criticality_index_delta_cr(_sample(_powerlaw(alpha), 40000, seed=0))
    assert abs(value) < THRESHOLD


@pytest.mark.parametrize("lam", [0.46, 0.30])
def test_exponential_is_subcritical(lam):
    """指数分布 (結合前のポアソン活動) は劣臨界: ΔCr が閾値より下。"""
    pmf = np.exp(-lam * np.arange(1, SMAX + 1))
    assert criticality_index_delta_cr(_sample(pmf, 20000, seed=1)) < -THRESHOLD


def test_bimodal_tail_is_supercritical():
    """大アバランシェのこぶ (同期バースト) は超臨界: ΔCr が正で閾値付近まで出る。"""
    bump = np.zeros(SMAX)
    bump[75:95] = 1.0
    pmf = 0.9 * _powerlaw(1.5) / _powerlaw(1.5).sum() + 0.1 * bump / bump.sum()
    assert criticality_index_delta_cr(_sample(pmf, 20000, seed=2)) > 0.15


def test_scale_stays_in_the_threshold_world():
    """ΔCr は点ごとのずれの「平均」。データ点数が増えても値のオーダーは変わらない。

    和にすると点数 (指数分布なら 20-30, べき乗なら 60-100) だけ値が膨らみ、
    Tetzlaff の ±0.195 という判定閾値が意味を失う。
    """
    pmf = np.exp(-0.46 * np.arange(1, SMAX + 1))
    for n in (5000, 50000):
        assert -1.0 < criticality_index_delta_cr(_sample(pmf, n, seed=n)) < -THRESHOLD


def test_too_few_points_is_nan():
    assert np.isnan(criticality_index_delta_cr(np.array([1, 2, 3])))
    assert np.isnan(criticality_index_delta_cr(np.array([], dtype=int)))


# ---------------------------------------------------------------------------
# LLR の打ち切り既定
# ---------------------------------------------------------------------------

from src.utils.analysis.powerlaw import log_likelihood_ratio_power_vs_exponential as _llr


def test_llr_default_is_untruncated():
    """既定は打ち切りなし。smax=100 を明示したときだけ 100 超が捨てられる。

    論文 Fig2(c) の LLR は [1,100] 打ち切りでは天井 (~0.48×N) に阻まれて再現できない。
    0h(純ポアソン, パラメータ自由度ゼロ) と 6h の両方を当てるのは離散・打ち切りなしだけ。
    """
    rng = np.random.default_rng(5)
    grid = np.arange(1, 501)
    pmf = grid.astype(float) ** -1.5
    sizes = rng.choice(grid, size=20000, p=pmf / pmf.sum())
    assert (sizes > 100).sum() > 100  # 100 超がちゃんとある標本
    assert _llr(sizes) > _llr(sizes, smax=100) * 1.5


def test_llr_truncated_has_a_ceiling():
    """[1,100] 打ち切りでは、完全なべき乗でも 1 アバランシェあたり 0.5 を超えない。"""
    rng = np.random.default_rng(6)
    grid = np.arange(1, SMAX + 1)
    n = 20000
    for alpha in (1.3, 1.6, 2.0):
        pmf = grid.astype(float) ** -alpha
        sizes = rng.choice(grid, size=n, p=pmf / pmf.sum())
        assert _llr(sizes, smax=SMAX) / n < 0.5


def test_llr_is_negative_for_exponential_sizes():
    """指数分布 (結合前のポアソン活動) では power-law が負けて LLR < 0。

    連続版 MLE はここで符号を誤る (s=1 の atom を密度で扱うため)。論文の 0h は負なので、
    この符号が論文の実装が離散である証拠になっている。
    """
    rng = np.random.default_rng(7)
    grid = np.arange(1, SMAX + 1)
    pmf = np.exp(-0.46 * grid)
    sizes = rng.choice(grid, size=9000, p=pmf / pmf.sum())
    assert _llr(sizes) < 0


def test_smin_is_calibrated_at_the_poisson_reference():
    """既定 smin=3 は 0h(結合ゼロ=純ポアソン)で論文値を再現するよう較正されている。

    0h のアバランシェサイズは幾何分布になる。論文 Fig 2(c) の 0h は ΔCr = -0.232
    (画素実測) で、smin=3 の指数分布サンプルはその近傍に落ちる。smin を大きく取ると
    2 倍過大になるので、この較正はスケールを固定する意味を持つ。
    """
    pmf = np.exp(-0.46 * np.arange(1, SMAX + 1))   # 平均ISI閾値のポアソン活動に相当
    value = criticality_index_delta_cr(_sample(pmf, 9000, seed=11))
    assert -0.30 < value < -0.18
    # smin を上げると系統的に過大評価になる (自動選択が踏んでいた罠)
    assert criticality_index_delta_cr(_sample(pmf, 9000, seed=11), smin=8) < value
