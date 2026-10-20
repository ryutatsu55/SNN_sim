"""離散値のべき乗分布フィット。

経験 PMF と、Clauset et al. (2009) / Yada et al. (2017) 準拠の離散打ち切り最尤推定
(power-law / exponential) およびその対数尤度比 (LLR) を提供する。

描画用の曲線もここで作る (`fit_distribution_curves`)。matplotlib を持ち込まずに
「経験分布に重ねる 2 本の理論曲線」まで確定させておくことで、描画側は線種と凡例だけを
決めればよくなる。

記号
----
s               アバランシェサイズ (スパイク数、1 以上の整数)。
[smin, smax]    フィットに使うサイズの範囲。**確率の台でもある** —— 2 つのモデルは
                どちらもこの範囲で正規化されるので、範囲を変えると尤度の絶対値が動く。
                アバランシェでは smin=1 / smax=N (系のニューロン数)。
α (alpha)       べき乗モデルの**指数**。p(s) = s^-α / Σ_{k=smin}^{smax} k^-α。
                正の値で、大きいほど裾が速く落ちる。臨界系では 1.5 付近。
                `alpha_mle` は最尤推定、`slope_loglog` は log-log 回帰の傾き (= -α)。
λ (lam)         指数モデルの**減衰率**。p(s) = e^(-λs) / Σ_{k=smin}^{smax} e^(-λk)。
                単位は 1/サイズで、特徴サイズ 1/λ を超えるとほぼ出なくなる。
                λ が大きいほど小さいアバランシェしか出ない = 劣臨界的。
LLR             Σ_i [ln p_pow(s_i) − ln p_exp(s_i)]。正ならべき乗モデルの方が
                その標本をよく説明する。**平均ではなく和**なのでアバランシェ数に比例する。
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


def _fit_discrete_powerlaw_alpha(x: np.ndarray, smin: int = 1, smax: int = 100) -> float:
    """離散打ち切り power-law p(s) = s^-α / Σ_{k=smin}^{smax} k^-α の α を最尤推定する。

    α は分布の**指数** (正値、大きいほど裾が速く落ちる)。探索範囲は [1.01, 6.0]。
    """
    support = np.arange(smin, smax + 1, dtype=np.float64)
    sum_log = float(np.sum(np.log(x)))
    n = x.size

    def neg_ll(alpha: float) -> float:
        norm = float(np.sum(np.power(support, -alpha)))
        return alpha * sum_log + n * np.log(norm)

    res = optimize.minimize_scalar(neg_ll, bounds=(1.01, 6.0), method="bounded")
    return float(res.x)


def _fit_discrete_exponential_lambda(x: np.ndarray, smin: int = 1, smax: int = 100) -> float:
    """離散打ち切り指数分布 p(s) = e^(-λs) / Σ_{k=smin}^{smax} e^(-λk) の λ を最尤推定する。

    λ は**減衰率** [1/サイズ] で、特徴サイズ 1/λ を超えるとほぼ出なくなる。
    探索範囲は [1e-6, 5.0]。
    """
    support = np.arange(smin, smax + 1, dtype=np.float64)
    sum_x = float(np.sum(x))
    n = x.size

    def neg_ll(lam: float) -> float:
        norm = float(np.sum(np.exp(-lam * support)))
        return lam * sum_x + n * np.log(norm)

    res = optimize.minimize_scalar(neg_ll, bounds=(1e-6, 5.0), method="bounded")
    return float(res.x)


def log_likelihood_ratio_power_vs_exponential(
    sizes: np.ndarray, smax: int | None = 100
) -> float:
    """power-law vs exponential の対数尤度比 (正 → power-law 優位)。

    サイズ [1, smax] で離散打ち切り power-law と exponential を最尤フィットし、
    LLR = Σ_i [ln p_power(s_i) − ln p_exp(s_i)] を返す。Clauset et al. (2009) /
    Yada et al. (2017) と同じ離散打ち切り MLE。

    **打ち切り範囲はフィット範囲そのもの。** Ikeda-Akita-Takahashi 2023 supplementary
    II.B は「系が 100 ニューロンなので、サイズ 1〜100 のアバランシェを fitting に使った」
    と書く。既定の `smax=100` はこれで、アバランシェ側の系サイズ N を渡すのが本筋。

    `smax=None` は打ち切らない (観測最大まで)。**論文の手順ではない**うえ、裾の数個の
    巨大アバランチに値が強く依存する。

    打ち切った LLR には天井がある —— 分布がべき乗である限り 1 アバランシェあたり
    `llr_ceiling_per_avalanche(smax)` (smax=100 で 0.48) を超えない。絶対値を他所の
    数字と比べる前に、まず天井の何割かを見ること。
    """
    smin = 1
    x = np.asarray(sizes, dtype=np.float64)
    x = x[x >= smin]
    if x.size < 2:
        return np.nan
    top = int(smax) if smax is not None else int(x.max())
    x = x[x <= top]
    if x.size < 2:
        return np.nan

    support = np.arange(smin, top + 1, dtype=np.float64)
    alpha = _fit_discrete_powerlaw_alpha(x, smin, top)
    lam = _fit_discrete_exponential_lambda(x, smin, top)

    log_norm_power = float(np.log(np.sum(np.power(support, -alpha))))
    log_norm_exp = float(np.log(np.sum(np.exp(-lam * support))))

    ll_power = -alpha * np.log(x) - log_norm_power
    ll_exp = -lam * x - log_norm_exp
    return float(np.sum(ll_power - ll_exp))


def llr_ceiling_per_avalanche(smax: int = 100, alpha_grid: np.ndarray | None = None) -> float:
    """[1, smax] のデータがべき乗であるとき、LLR が 1 アバランシェあたり取りうる最大値。

    Args:
        smax: 確率の台の上限。天井はこれに依存する (台が広いほどべき乗と指数の差が出る)。
        alpha_grid: 最大値を探す**べき指数 α の候補**。フィットではなく走査で、
            「どの α のべき乗が指数分布から最も遠いか」を探すためのもの。
            既定は 1.05〜3.0 を 0.01 刻み。λ 側も内部で 1e-4〜2.0 を走査する。

    LLR/N は大数の法則で
        min_λ KL(q ‖ Exp_λ) − min_α KL(q ‖ PL_α)
    に収束する。データが真にべき乗 (q = PL_α) なら第2項は 0 なので、上限は
        max_α [ min_λ KL(PL_α ‖ Exp_λ) ]
    で、smax=100 では **α≈1.6 の 0.49 nats** になる。標本を作らず決定論的に計算する。

    **これは 1 アバランシェあたりの量。** LLR は和なので、比べる相手は
    `天井 × アバランシェ数`。したがって「LLR が大きい」だけでは何も言えない ——
    アバランシェ数が多い run はそれだけで大きな LLR を出すし、smax が広いほど
    天井自体も上がる (smax=100 で 0.480、256 で 0.756、500 で 0.986)。
    **見るべきは `LLR / アバランシェ数` が天井の何割か。**

    これが要るのは、論文 Fig 2(c) の LLR が [1,100] の天井を超えて見えるため。
    72h の 19862 が天井の内側に収まるにはアバランシェ数が 41353 個以上必要だが、
    **論文はアバランシェ数を書いていない** ——
    こちらの 72h (発火率 1.12/3.31 Hz は論文の ~1.2/~3.0 とほぼ同じ) は
    93373 spikes / 23839 アバランシェなので、同じ発火率なら 41353 には届かない
    (1.7 倍の発火が要る)。**分母は推定値**であることを承知で読むこと。
    打ち切りを外せば台が広がって天井も上がるので、この矛盾は消える。
    """
    support = np.arange(1, smax + 1, dtype=np.float64)
    if alpha_grid is None:
        alpha_grid = np.arange(1.05, 3.0, 0.01)
    lam_grid = np.arange(1e-4, 2.0, 1e-3)
    log_exp = -np.outer(lam_grid, support)
    log_exp -= np.log(np.exp(log_exp).sum(axis=1))[:, None]

    best = -np.inf
    for alpha in alpha_grid:
        q = support ** -alpha
        q /= q.sum()
        log_q = np.log(q)
        # min_λ KL(q‖Exp_λ) = -H(q) - max_λ Σ q ln p_λ
        value = float(np.sum(q * log_q) - np.max(log_exp @ q))
        best = max(best, value)
    return best


def fit_exponent(values: np.ndarray, xmin: int = 1, xmax: int = 100) -> dict[str, float]:
    """べき乗指数を最尤推定 + log-log 回帰の両方で求める。

    Returns:
        alpha_mle   : 離散打ち切り MLE の指数 (正値。p(x) ∝ x^-alpha)
        slope_loglog: log-log 平面での最小二乗回帰の傾き (負値。論文の α に対応)
        llr         : power-law vs exponential の対数尤度比 (正 → power-law 優位)。
                      指数と同じ [xmin, xmax] で評価する。
        num_samples : 指数の推定に使ったサンプル数 ([xmin, xmax] 内)
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
    # 指数・回帰傾き・LLR のすべてが同じ [xmin, xmax] のもの。論文はフィット範囲を
    # 系サイズで切ると書いており、尤度だけ別範囲で評価する理由が無い。
    out["llr"] = log_likelihood_ratio_power_vs_exponential(values, smax=xmax)

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

    # fit_exponent には**切っていない** values を渡す。指数・回帰傾き・LLR はどれも内部で
    # [1, fit_max] に切って評価されるので、図の凡例と metrics.csv が一致する。
    fit = fit_exponent(values, xmin=1, xmax=fit_max)
    data = np.asarray(values, dtype=np.float64)
    data = data[(data >= 1) & (data <= fit_max)]

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
