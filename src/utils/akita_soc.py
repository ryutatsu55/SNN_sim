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


def avalanche_distribution(sizes: np.ndarray, smax: int = 100) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(sizes, dtype=np.int32)
    valid = valid[(valid >= 1) & (valid <= smax)]
    if valid.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)
    counts = np.bincount(valid, minlength=smax + 1)[1:]
    support = np.arange(1, smax + 1)
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


def plot_raster(times: np.ndarray, ids: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(times / 1000.0, ids, s=2.0, color="black")
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Neuron ID")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_avalanche_distribution(sizes: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    support, prob = avalanche_distribution(sizes)
    fig, ax = plt.subplots(figsize=(5, 4))
    if support.size > 0:
        ax.scatter(support, prob, s=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Avalanche size")
    ax.set_ylabel("Probability")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
