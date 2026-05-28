from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    x = np.log(support)
    y = np.log(prob)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = np.exp(intercept) * np.power(support, slope)
    fitted = fitted / np.sum(fitted)
    return slope, intercept, fitted


def criticality_index_delta_cr(sizes: np.ndarray, smax: int = 100) -> float:
    support, prob = avalanche_distribution(sizes, smax=smax)
    if support.size < 3:
        return np.nan

    best_error = np.inf
    best_support = support
    best_prob = prob
    best_fit = prob
    for smin in support[:-2]:
        mask = support >= smin
        if np.count_nonzero(mask) < 3:
            continue
        _, _, fit = _fit_power_loglog(support[mask], prob[mask])
        error = float(np.sum((np.log(prob[mask]) - np.log(fit)) ** 2))
        if error < best_error:
            best_error = error
            best_support = support[mask]
            best_prob = prob[mask]
            best_fit = fit

    diff = best_prob - best_fit
    upper = float(np.sum(np.maximum(diff, 0.0)))
    lower = float(np.sum(np.minimum(diff, 0.0)))
    return upper if abs(upper) >= abs(lower) else lower


def log_likelihood_ratio_power_vs_exponential(sizes: np.ndarray, smax: int = 100) -> float:
    """離散サイズ1..smax上で、power-lawとexponentialの簡易LLRを計算する。"""
    x = np.asarray(sizes, dtype=np.float64)
    x = x[(x >= 1) & (x <= smax)]
    if x.size < 2:
        return np.nan

    log_sum = float(np.sum(np.log(x)))
    if log_sum <= 0.0:
        return np.nan
    alpha = 1.0 + (x.size / log_sum)
    support = np.arange(1, smax + 1, dtype=np.float64)
    p_power = np.power(support, -alpha)
    p_power /= np.sum(p_power)

    lam = 1.0 / max(np.mean(x) - 1.0, 1e-12)
    p_exp = np.exp(-lam * (support - 1.0))
    p_exp /= np.sum(p_exp)

    indices = x.astype(np.int32) - 1
    return float(np.sum(np.log(p_power[indices]) - np.log(p_exp[indices])))


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


def plot_raster(
    times: np.ndarray,
    ids: np.ndarray,
    out_path: Path,
    title: str,
    xlim_s: tuple[float, float] | None = None,
    ylim_neuron: tuple[float, float] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(times / 1000.0, ids, s=2.0, color="black")
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Neuron ID")
    if xlim_s is not None:
        ax.set_xlim(*xlim_s)
    if ylim_neuron is not None:
        ax.set_ylim(*ylim_neuron)
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
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    support, prob = avalanche_distribution(sizes, smax=smax)
    fig, ax = plt.subplots(figsize=(5, 4))
    if support.size > 0:
        ax.scatter(support, prob, s=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
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
