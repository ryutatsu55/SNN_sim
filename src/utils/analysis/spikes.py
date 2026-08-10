"""スパイク列から得られる発火量の集計と、その書き出し。"""
from __future__ import annotations

from pathlib import Path

import numpy as np


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


def export_spike_csv(spike_time: np.ndarray, neuron_id: np.ndarray, output_path="spikes.csv"):
    """発火時刻とニューロンIDをCSVとして保存する関数"""

    spike_time = np.asarray(spike_time, dtype=float).reshape(-1)
    neuron_id = np.asarray(neuron_id).reshape(-1)

    if spike_time.size == 0 or neuron_id.size == 0:
        raise ValueError("spike_time and neuron_id arrays must not be empty.")
    if spike_time.shape[0] != neuron_id.shape[0]:
        raise ValueError("spike_time and neuron_id arrays must have the same length.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spike_table = np.column_stack((spike_time, neuron_id.astype(int, copy=False)))
    np.savetxt(
        output_path,
        spike_table,
        delimiter=",",
        header="spike_time,neuron_id",
        comments="",
        fmt=["%.10f", "%d"],
    )
