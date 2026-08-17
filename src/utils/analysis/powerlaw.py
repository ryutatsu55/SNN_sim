"""離散値のべき乗分布フィット。

経験 PMF と、Clauset et al. (2009) / Yada et al. (2017) 準拠の離散打ち切り最尤推定
(power-law / exponential) およびその対数尤度比 (LLR) を提供する。

描画用の曲線もここで作る (`fit_distribution_curves`)。matplotlib を持ち込まずに
「経験分布に重ねる 2 本の理論曲線」まで確定させておくことで、描画側は線種と凡例だけを
決めればよくなる。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


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


def _fit_power_loglog(support: np.ndarray, prob: np.ndarray) -> tuple[float, float, np.ndarray]:
    """log-log 平面での power-law 線形回帰。

    pfit(s) = exp(intercept)·s^slope の**生の回帰線**を返す(合計1へ再正規化しない)。
    ΔCr(S21-24)は pemp と pfit の差の符号で sub/super を判定するため、pfit を PMF へ
    再正規化すると Σ(pemp−pfit)=0 となり Aupper=|Alower| で符号情報が消えてしまう。
    """
    x = np.log(support)
    y = np.log(prob)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = np.exp(intercept) * np.power(support, slope)
    return slope, intercept, fitted


def _fit_discrete_powerlaw_alpha(x: np.ndarray, smin: int = 1, smax: int = 100) -> float:
    """離散打ち切りpower-law p(s)=s^-alpha / sum_{k=smin}^{smax} k^-alpha のMLE指数。"""
    support = np.arange(smin, smax + 1, dtype=np.float64)
    sum_log = float(np.sum(np.log(x)))
    n = x.size

    def neg_ll(alpha: float) -> float:
        norm = float(np.sum(np.power(support, -alpha)))
        return alpha * sum_log + n * np.log(norm)

    res = optimize.minimize_scalar(neg_ll, bounds=(1.01, 6.0), method="bounded")
    return float(res.x)


def _fit_discrete_exponential_lambda(x: np.ndarray, smin: int = 1, smax: int = 100) -> float:
    """離散打ち切り指数分布 p(s)=e^-lambda*s / sum_{k=smin}^{smax} e^-lambda*k のMLE率。"""
    support = np.arange(smin, smax + 1, dtype=np.float64)
    sum_x = float(np.sum(x))
    n = x.size

    def neg_ll(lam: float) -> float:
        norm = float(np.sum(np.exp(-lam * support)))
        return lam * sum_x + n * np.log(norm)

    res = optimize.minimize_scalar(neg_ll, bounds=(1e-6, 5.0), method="bounded")
    return float(res.x)


def log_likelihood_ratio_power_vs_exponential(sizes: np.ndarray, smax: int = 100) -> float:
    """論文準拠のLLR: サイズ[1, smax]でpower-lawとexponentialを最尤フィットし、
    対数尤度比 LLR = sum_i [ln p_power(s_i) - ln p_exp(s_i)] を返す (正→power-law優位)。

    Clauset et al. (2009) / Yada et al. (2017) と同じ離散打ち切りMLEを用いる
    (Akita Supplementary "Power-law fitting", smin=1, smax=100)。
    """
    smin = 1
    x = np.asarray(sizes, dtype=np.float64)
    x = x[(x >= smin) & (x <= smax)]
    if x.size < 2:
        return np.nan

    support = np.arange(smin, smax + 1, dtype=np.float64)
    alpha = _fit_discrete_powerlaw_alpha(x, smin, smax)
    lam = _fit_discrete_exponential_lambda(x, smin, smax)

    log_norm_power = float(np.log(np.sum(np.power(support, -alpha))))
    log_norm_exp = float(np.log(np.sum(np.exp(-lam * support))))

    ll_power = -alpha * np.log(x) - log_norm_power
    ll_exp = -lam * x - log_norm_exp
    return float(np.sum(ll_power - ll_exp))


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


@dataclass
class DistributionFit:
    """経験 PMF と、それに重ねるべき乗/指数の理論曲線。

    `powerlaw` / `exponential` は `fit_support` (= 1..fit_max) 上で評価し、**経験分布の
    その範囲内の確率質量にスケールしてある**ので、経験 PMF とそのまま重ね描きできる。
    フィット不能 (サンプル 2 個未満) のときは両方とも空配列で、指標は nan。
    """
    support: np.ndarray       # 経験 PMF の support (値が現れた点のみ)
    prob: np.ndarray          # 経験 PMF
    fit_support: np.ndarray   # 1..fit_max
    powerlaw: np.ndarray      # スケール済みのべき乗曲線
    exponential: np.ndarray   # スケール済みの指数曲線
    alpha: float              # 離散打ち切り MLE の指数 (正値)
    lam: float                # 離散打ち切り MLE の率
    llr: float                # power-law vs exponential の対数尤度比
    slope_loglog: float       # log-log 回帰の傾き (負値)
    num_fitted: int           # フィットに使ったサンプル数


def fit_distribution_curves(
    values: np.ndarray,
    fit_max: int = 100,
    support_max: int | None = None,
) -> DistributionFit:
    """経験 PMF を作り、[1, fit_max] で power-law / exponential を最尤フィットする。

    Args:
        values: 正の整数値の標本 (アバランシェサイズ、寿命ビン数など)
        fit_max: 最尤フィットの上限。理論曲線もこの範囲で描く。
        support_max: 経験 PMF 側の打ち切り。None なら打ち切らない (fit_max より大きい
            サイズも散布図には出る)。
    """
    support, prob = discrete_distribution(values, xmax=support_max)
    fit_support = np.arange(1, fit_max + 1, dtype=np.float64)
    empty = np.array([], dtype=np.float64)

    data = np.asarray(values, dtype=np.float64)
    data = data[(data >= 1) & (data <= fit_max)]
    fit = fit_exponent(data, xmin=1, xmax=fit_max)

    if support.size == 0 or data.size < 2:
        return DistributionFit(
            support=support, prob=prob, fit_support=fit_support,
            powerlaw=empty, exponential=empty,
            alpha=fit["alpha_mle"], lam=float("nan"), llr=fit["llr"],
            slope_loglog=fit["slope_loglog"], num_fitted=int(data.size),
        )

    lam = _fit_discrete_exponential_lambda(data, 1, fit_max)
    # 経験分布の [1, fit_max] 内の確率質量に合わせると、表示スケールが揃う。
    empirical_mass = float(prob[support <= fit_max].sum())
    powerlaw = np.power(fit_support, -fit["alpha_mle"])
    powerlaw = powerlaw / powerlaw.sum() * empirical_mass
    exponential = np.exp(-lam * fit_support)
    exponential = exponential / exponential.sum() * empirical_mass

    return DistributionFit(
        support=support, prob=prob, fit_support=fit_support,
        powerlaw=powerlaw, exponential=exponential,
        alpha=fit["alpha_mle"], lam=lam, llr=fit["llr"],
        slope_loglog=fit["slope_loglog"], num_fitted=int(data.size),
    )
