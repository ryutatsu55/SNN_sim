"""ΔCr (Ikeda-Akita-Takahashi 2023 S21-S24) と LLR の性質を固定する。

ΔCr は「経験分布と、log-log 平面に引いた回帰直線が与えるべき乗との、**確率の差の和**」。
差を log 空間で取ると回帰と同じ平面になり、最小二乗の性質から Σ残差 = 0、すなわち
Aupper = |Alower| となって符号が丸め誤差で決まる。

回帰は**観測個数で重み付ける**。log10(pemp) の分散は 1/count に比例するので、等重みだと
数個しかない裾に直線が引きずられ、小サイズ側で pfit が確率 1 を超えて和が数単位まで
膨らむ。ここで縛るのは (a) べき乗で ~0、劣臨界で負、(b) 値の尺度が標本数で動かないこと、
(c) 回帰直線が確率 1 を超えないこと。

**既知の限界**: 大サイズのこぶ (超臨界) は「0 から離れる」が、この定義では符号が正に
ならない。論文 Fig 2(c) の値も超臨界とされる 6h で負 (−0.042) で、正になるのは
+0.017 が最大。符号ではなく**時間発展**で読む指標として扱うこと。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.analysis.criticality import criticality_index_delta_cr, delta_cr_fit

SMAX = 100


def _sample(pmf: np.ndarray, n: int, seed: int) -> np.ndarray:
    grid = np.arange(1, SMAX + 1)
    return np.random.default_rng(seed).choice(grid, size=n, p=pmf / pmf.sum())


def _powerlaw(alpha: float) -> np.ndarray:
    return np.arange(1, SMAX + 1, dtype=float) ** -alpha


def _exponential(lam: float) -> np.ndarray:
    return np.exp(-lam * np.arange(1, SMAX + 1))


def _bimodal() -> np.ndarray:
    bump = np.zeros(SMAX)
    bump[75:95] = 1.0
    return 0.9 * _powerlaw(1.5) / _powerlaw(1.5).sum() + 0.1 * bump / bump.sum()


@pytest.mark.parametrize("alpha", [1.5, 1.7, 2.0])
def test_pure_power_law_is_critical(alpha):
    """べき乗そのものは臨界: ΔCr が 0 の近く。"""
    assert abs(criticality_index_delta_cr(_sample(_powerlaw(alpha), 40000, seed=0))) < 0.05


@pytest.mark.parametrize("lam", [0.46, 0.30])
def test_exponential_is_subcritical(lam):
    """指数分布 (結合前のポアソン活動) は劣臨界: ΔCr が負。"""
    assert criticality_index_delta_cr(_sample(_exponential(lam), 20000, seed=1)) < -0.3


def test_deviation_is_larger_than_for_a_power_law():
    """べき乗から外れた分布は、べき乗より 0 から遠い。

    大サイズのこぶ (超臨界) は**正にはならない** —— 重み付き回帰は小サイズに固定され、
    こぶが傾きを寝かせて中間サイズが直線の下に潜るため。ここで固定するのは
    「外れていることを検出する」ことだけ。
    """
    critical = abs(criticality_index_delta_cr(_sample(_powerlaw(1.5), 20000, seed=3)))
    for pmf in (_exponential(0.46), _bimodal()):
        assert abs(criticality_index_delta_cr(_sample(pmf, 20000, seed=3))) > 5 * critical


def test_the_regression_stays_a_probability():
    """回帰直線が s=1 で確率 1 を超えないこと。

    等重みの最小二乗はここで 2.9 まで跳ね上がっていた (論文 Fig S1 の回帰線は
    s=1 で ~0.8)。跳ね上がると ΔCr の絶対値がそのぶん丸ごと膨らむ。
    """
    for pmf in (_powerlaw(1.5), _exponential(0.46), _exponential(0.10), _bimodal()):
        assert delta_cr_fit(_sample(pmf, 20000, seed=1)).fit[0] < 1.0


def test_the_scale_does_not_drift_with_sample_size():
    """標本数を 25 倍にしても値の尺度が動かないこと。

    等重みだと裾の点が増えるほど直線が寝て、同じ分布でも −1.4 → −4.8 と膨らんでいた。
    """
    values = [criticality_index_delta_cr(_sample(_exponential(0.46), n, seed=n))
              for n in (2000, 50000)]
    assert max(values) / min(values) < 1.5


@pytest.mark.parametrize("seed", [11, 12, 13, 14, 15])
def test_the_sign_does_not_depend_on_the_sample(seed):
    """劣臨界の符号が標本ごとに反転しないこと。

    回帰と同じ log 空間で残差を取ると Aupper = |Alower| となり、どちらを返すかが
    1e-16 の丸めで決まっていた (同じ指数分布で λ を変えただけで符号が反転した)。
    確率空間の差にはこの縮退が無い。
    """
    assert criticality_index_delta_cr(_sample(_exponential(0.46), 9000, seed=seed)) < 0.0


def test_lower_cutoff_shrinks_the_deviation():
    """`smin` は明示指定のときだけ効く。上げるほど小サイズのずれが落ちる。

    論文は smin を「線形フィットの二乗誤差和が最小になるよう決めた」と書くが、その規準は
    点を減らすほど誤差が減るので走査範囲の端に張り付く。既定 (=1) は切らない。
    """
    sizes = _sample(_exponential(0.46), 9000, seed=11)
    assert (criticality_index_delta_cr(sizes)
            < criticality_index_delta_cr(sizes, smin=3)
            < 0.0)


def test_too_few_points_is_nan():
    assert np.isnan(criticality_index_delta_cr(np.array([1, 2, 3])))
    assert np.isnan(criticality_index_delta_cr(np.array([], dtype=int)))


# ---------------------------------------------------------------------------
# LLR の打ち切り
# ---------------------------------------------------------------------------

from src.utils.analysis.powerlaw import log_likelihood_ratio_power_vs_exponential as _llr


def test_llr_truncates_at_the_fit_range_by_default():
    """既定は [1, 100] 打ち切り (論文 supplementary II.B の fitting 範囲)。

    `smax=None` は打ち切らない = 論文の手順ではない。裾の巨大アバランシェが入るぶん
    値が大きく出る。
    """
    rng = np.random.default_rng(5)
    grid = np.arange(1, 501)
    pmf = grid.astype(float) ** -1.5
    sizes = rng.choice(grid, size=20000, p=pmf / pmf.sum())
    assert (sizes > 100).sum() > 100  # 100 超がちゃんとある標本
    assert _llr(sizes) == _llr(sizes, smax=100)
    assert _llr(sizes, smax=None) > _llr(sizes) * 1.5


def test_llr_truncated_has_a_ceiling():
    """[1, 100] 打ち切りでは、完全なべき乗でも 1 アバランシェあたり 0.5 を超えない。"""
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
