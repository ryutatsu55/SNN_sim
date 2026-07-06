from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


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


def avalanche_distribution(sizes: np.ndarray, smax: int | None = 100) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(sizes, dtype=np.int32)
    valid = valid[valid >= 1]
    if smax is not None:
        valid = valid[valid <= smax]
    if valid.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
    max_size = int(np.max(valid)) if smax is None else int(smax)
    counts = np.bincount(valid, minlength=max_size + 1)[1:]
    support = np.arange(1, max_size + 1)
    mask = counts > 0
    return support[mask], counts[mask] / valid.size


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


def criticality_index_delta_cr(
    sizes: np.ndarray, smax: int = 100, smin_max: int = 10, min_points: int = 10
) -> float:
    """臨界性指標 ΔCr(Ikeda-Akita-Takahashi 2023 supplementary 式 S21-S24)。

    pemp(s)=経験PMF, pfit(s)=log-log 線形回帰の**生の回帰線**(exp(b)·s^slope)。
    smin を [1, smin_max] で走査して線形回帰の平均二乗誤差を最小化(論文: 小サイズは
    べき乗からずれるため除外)。smax=100。
        Aupper = Σ max(pemp−pfit, 0),  Alower = Σ min(pemp−pfit, 0)   [smin..smax]
        ΔCr    = |大きい方の符号付き値|。 正=超臨界, ≈0=臨界, 負=劣臨界。

    重要: pfit は**再正規化しない**(旧実装は合計1へ正規化していたため Σ(pemp−pfit)=0 と
    なり、かつ smin>1 選択時に −(小サイズ質量) の系統誤差が入って偽の強い劣臨界値を出していた)。
    smin は小範囲に制限(旧実装は上限なしで tail へ退化)。
    """
    support, prob = avalanche_distribution(sizes, smax=smax)
    if support.size < min_points:
        return np.nan

    best_error = np.inf
    best_prob = None
    best_fit = None
    for smin in support:
        if smin > smin_max:
            break
        mask = support >= smin
        if np.count_nonzero(mask) < min_points:
            break
        _, _, fit = _fit_power_loglog(support[mask], prob[mask])
        error = float(np.mean((np.log(prob[mask]) - np.log(fit)) ** 2))
        if error < best_error:
            best_error = error
            best_prob = prob[mask]
            best_fit = fit

    if best_prob is None:
        return np.nan

    diff = best_prob - best_fit
    upper = float(np.sum(np.maximum(diff, 0.0)))
    lower = float(np.sum(np.minimum(diff, 0.0)))
    return upper if abs(upper) >= abs(lower) else lower


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


def firing_rates(spike_ids: np.ndarray, num_neurons: int, duration_ms: float) -> np.ndarray:
    if duration_ms <= 0:
        return np.zeros(num_neurons, dtype=np.float64)
    counts = np.bincount(np.asarray(spike_ids, dtype=np.int32), minlength=num_neurons)
    return counts[:num_neurons] / (duration_ms / 1000.0)


def spike_group_metrics(
    spike_ids: np.ndarray,
    excitatory_ids: np.ndarray,
    inhibitory_ids: np.ndarray,
    duration_ms: float,
) -> dict[str, float | int]:
    """グローバルIDのスパイク列からE/I別発火メトリクスを計算する。"""
    ids = np.asarray(spike_ids, dtype=np.int32)
    exc_ids = np.asarray(excitatory_ids, dtype=np.int32)
    inh_ids = np.asarray(inhibitory_ids, dtype=np.int32)
    duration_s = duration_ms / 1000.0

    exc_spikes = int(np.isin(ids, exc_ids).sum())
    inh_spikes = int(np.isin(ids, inh_ids).sum())
    if duration_s <= 0.0:
        exc_rate = np.nan
        inh_rate = np.nan
    else:
        exc_rate = exc_spikes / (max(exc_ids.size, 1) * duration_s)
        inh_rate = inh_spikes / (max(inh_ids.size, 1) * duration_s)

    return {
        "exc_spikes": exc_spikes,
        "inh_spikes": inh_spikes,
        "exc_rate_hz": float(exc_rate),
        "inh_rate_hz": float(inh_rate),
    }


def _weight_block_stats(block: np.ndarray, wmax: float, at_max_tolerance: float) -> dict[str, float]:
    block = np.asarray(block, dtype=np.float64)
    if block.size == 0:
        return {
            "mean": np.nan,
            "nonzero_fraction": np.nan,
            "at_max_fraction": np.nan,
        }
    return {
        "mean": float(np.mean(block)),
        "nonzero_fraction": float(np.mean(block > 0.0)),
        "at_max_fraction": float(np.mean(block >= (wmax - at_max_tolerance))),
    }


def weight_block_metrics(
    weights: np.ndarray,
    excitatory_ids: np.ndarray,
    inhibitory_ids: np.ndarray,
    wmax: float,
    connection_mask: np.ndarray | None = None,
    at_max_tolerance: float = 1e-3,
) -> dict[str, float]:
    """グローバル重み行列から全体およびE/Iブロック別の統計を計算する。"""
    matrix = np.asarray(weights, dtype=np.float64)
    mask = None if connection_mask is None else np.asarray(connection_mask) != 0
    exc_ids = np.asarray(excitatory_ids, dtype=np.int32)
    inh_ids = np.asarray(inhibitory_ids, dtype=np.int32)

    all_values = matrix[mask] if mask is not None else matrix
    all_stats = _weight_block_stats(all_values, wmax=wmax, at_max_tolerance=at_max_tolerance)
    metrics = {
        "weight_mean": all_stats["mean"],
        "weight_nonzero_fraction": all_stats["nonzero_fraction"],
        "weight_at_max_fraction": all_stats["at_max_fraction"],
    }

    blocks = {
        "ee": (exc_ids, exc_ids),
        "ei": (exc_ids, inh_ids),
        "ie": (inh_ids, exc_ids),
        "ii": (inh_ids, inh_ids),
    }
    for name, (source_ids, target_ids) in blocks.items():
        block = matrix[np.ix_(source_ids, target_ids)]
        if mask is not None:
            block_mask = mask[np.ix_(source_ids, target_ids)]
            block = block[block_mask]
        stats = _weight_block_stats(block, wmax=wmax, at_max_tolerance=at_max_tolerance)
        metrics[f"weight_{name}_mean"] = stats["mean"]
        metrics[f"weight_{name}_at_max_fraction"] = stats["at_max_fraction"]

    return metrics


def diagnose_activity(
    mean_rate_hz: float,
    weight_at_max_fraction: float,
    overactive_rate_hz: float = 20.0,
    weight_saturation_fraction: float = 0.5,
) -> dict[str, bool | str]:
    """平均発火率と重み飽和率から実験状態を簡易診断する。"""
    is_overactive = bool(np.isfinite(mean_rate_hz) and mean_rate_hz >= overactive_rate_hz)
    is_weight_saturated = bool(
        np.isfinite(weight_at_max_fraction)
        and weight_at_max_fraction >= weight_saturation_fraction
    )

    if is_overactive and is_weight_saturated:
        diagnosis = "overactive_and_weight_saturated"
    elif is_overactive:
        diagnosis = "overactive"
    elif is_weight_saturated:
        diagnosis = "weight_saturated"
    else:
        diagnosis = "ok"

    return {
        "is_overactive": is_overactive,
        "is_weight_saturated": is_weight_saturated,
        "diagnosis": diagnosis,
    }


def build_group_display_map(
    group_ids: dict[str, np.ndarray],
) -> tuple[dict[int, int], int, int]:
    """グローバルニューロンID -> 表示ID の写像を構築する。

    重み分布の可視化と同様にニューロングループ順に並べ替える:
    興奮性を先頭ブロック (1..Nexc)、抑制性を後続ブロック (Nexc+1..Nexc+Ninh) に配置する。
    各グループ内はグローバルIDの昇順。

    Returns: (mapping, n_exc, n_inh)
    """
    exc = np.sort(np.asarray(group_ids.get("excitatory", []), dtype=np.int64))
    inh = np.sort(np.asarray(group_ids.get("inhibitory", []), dtype=np.int64))
    mapping: dict[int, int] = {}
    for disp, gid in enumerate(exc, start=1):
        mapping[int(gid)] = disp
    for disp, gid in enumerate(inh, start=len(exc) + 1):
        mapping[int(gid)] = disp
    return mapping, int(exc.size), int(inh.size)


def plot_raster(
    times: np.ndarray,
    ids: np.ndarray,
    out_path: Path,
    title: str,
    xlim_s: tuple[float, float] | None = None,
    ylim_neuron: tuple[float, float] | None = None,
    group_ids: dict[str, np.ndarray] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))

    remapped = False
    if group_ids is not None:
        mapping, n_exc, n_inh = build_group_display_map(group_ids)
        if mapping:
            disp = np.array([mapping.get(int(i), -1) for i in ids], dtype=np.int64)
            valid = disp >= 0
            t_s = times[valid] / 1000.0
            d = disp[valid]
            exc_mask = d <= n_exc
            ax.scatter(t_s[exc_mask], d[exc_mask], s=2.0, color="tab:red", label="Excitatory")
            ax.scatter(t_s[~exc_mask], d[~exc_mask], s=2.0, color="tab:blue", label="Inhibitory")
            if n_exc > 0 and n_inh > 0:
                ax.axhline(n_exc + 0.5, color="gray", lw=0.8, ls="--")
                ax.legend(loc="upper right", markerscale=3, fontsize=8)
            ax.set_ylim(0.5, n_exc + n_inh + 0.5)
            remapped = True

    if not remapped:
        ax.scatter(times / 1000.0, ids, s=2.0, color="black")
        if ylim_neuron is not None:
            ax.set_ylim(*ylim_neuron)

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Neuron ID (excitatory 1..Nexc, inhibitory above)")
    if xlim_s is not None:
        ax.set_xlim(*xlim_s)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_avalanche_distribution(
    sizes: np.ndarray,
    out_path: Path,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    smax: int | None = None,
    fit_smax: int = 100,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    support, prob = avalanche_distribution(sizes, smax=smax)
    fig, ax = plt.subplots(figsize=(5, 4))
    if support.size > 0:
        ax.scatter(support, prob, s=12, color="black", label="empirical", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")

        # LLR と同じ [1, fit_smax] の離散打ち切りMLEで power-law / exponential を推定し重ねる。
        # 経験分布の [1, fit_smax] 内の確率質量にスケールして表示スケールを揃える。
        x = np.asarray(sizes, dtype=np.float64)
        x = x[(x >= 1) & (x <= fit_smax)]
        if x.size >= 2:
            fit_support = np.arange(1, fit_smax + 1, dtype=np.float64)
            alpha = _fit_discrete_powerlaw_alpha(x, 1, fit_smax)
            lam = _fit_discrete_exponential_lambda(x, 1, fit_smax)
            emp_mass = float(prob[support <= fit_smax].sum())
            p_pow = np.power(fit_support, -alpha)
            p_pow = p_pow / p_pow.sum() * emp_mass
            p_exp = np.exp(-lam * fit_support)
            p_exp = p_exp / p_exp.sum() * emp_mass
            llr = log_likelihood_ratio_power_vs_exponential(sizes, smax=fit_smax)
            ax.plot(fit_support, p_pow, color="tab:red", lw=1.5,
                    label=f"power-law MLE (α={alpha:.2f})", zorder=2)
            ax.plot(fit_support, p_exp, color="tab:blue", lw=1.5, ls="--",
                    label=f"exponential MLE (λ={lam:.3f})", zorder=2)
            ax.legend(fontsize=6.5, loc="lower left", title=f"LLR={llr:.0f}", title_fontsize=6.5)
    elif xlim is not None or ylim is not None:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Avalanche size")
    ax.set_ylabel("Probability")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
