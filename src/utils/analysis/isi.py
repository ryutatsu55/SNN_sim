"""スパイク間隔 (ISI) の構造。

`spikes.py` が「単位時間あたり何発か」= 発火**量**を扱うのに対し、こちらは
**間隔の分布形**を扱う。同じ発火率でも規則的に打つのか束で打つのかは別の量で、
損傷からの回復では後者のほうが先に動くことがある。

すべて COO 的な平坦配列 (`times`, `ids`) を受け取り、matplotlib も src/models も
import しない。
"""
from __future__ import annotations

import numpy as np

# CV / Lv を定義するのに要る最小スパイク数。
# CV は間隔が 2 本以上 (= スパイク 3 発)、Lv は隣接する間隔の対が要るので同じく 3 発。
MIN_SPIKES_FOR_INTERVALS = 3


def isi_per_neuron(times: np.ndarray, ids: np.ndarray, num_neurons: int):
    """ニューロンごとの ISI を ragged 配列 (values, offsets) で返す。

    ニューロン i の ISI は `values[offsets[i]:offsets[i+1]]`。npz に保存できるよう、
    object 配列ではなく 2 本の平坦配列にしてある。

    Returns:
        (values: float64 (総間隔数,), offsets: int64 (num_neurons+1,))
    """
    times = np.asarray(times, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    if times.size != ids.size:
        raise ValueError(f"times ({times.size}) と ids ({ids.size}) の長さが違います。")

    # (id, time) でソートすれば、同じニューロンのスパイクが時刻順に固まる。
    order = np.lexsort((times, ids))
    sorted_ids, sorted_times = ids[order], times[order]

    counts = np.bincount(sorted_ids, minlength=num_neurons)[:num_neurons]
    spike_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    diffs = np.diff(sorted_times)
    # ニューロンをまたぐ差分を落とす。境界は各ニューロンの最後のスパイクの位置。
    same = np.diff(sorted_ids) == 0
    values = diffs[same] if diffs.size else np.array([], dtype=np.float64)

    # 間隔数は「スパイク数 - 1」(0 発なら 0)
    interval_counts = np.maximum(counts - 1, 0)
    offsets = np.concatenate([[0], np.cumsum(interval_counts)]).astype(np.int64)
    del spike_offsets
    return values.astype(np.float64), offsets


def _per_neuron_stat(values: np.ndarray, offsets: np.ndarray, fn) -> np.ndarray:
    """ragged な ISI に per-neuron の統計量を適用する。定義できなければ NaN。"""
    n = offsets.size - 1
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        chunk = values[offsets[i]:offsets[i + 1]]
        if chunk.size >= MIN_SPIKES_FOR_INTERVALS - 1:
            out[i] = fn(chunk)
    return out


def cv_isi(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """ニューロンごとの ISI の変動係数 (std/mean)。ポアソンで 1、規則的で 0。"""
    def _cv(chunk):
        mean = float(np.mean(chunk))
        return float(np.std(chunk) / mean) if mean > 0.0 else np.nan
    return _per_neuron_stat(values, offsets, _cv)


def local_variation(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """ニューロンごとの Local Variation Lv (Shinomoto et al. 2003)。

        Lv = 3/(n-1) * Σ ((I_k - I_{k+1}) / (I_k + I_{k+1}))^2

    **発火率が非定常でも使える**のが CV との違い。損傷直後は発火率そのものが動くので、
    「打ち方が変わったのか、量だけ変わったのか」を分けるにはこちらを見る。
    ポアソンで 1、規則的で 0、束状で > 1。
    """
    def _lv(chunk):
        if chunk.size < 2:
            return np.nan
        a, b = chunk[:-1], chunk[1:]
        denom = a + b
        valid = denom > 0.0
        if not np.any(valid):
            return np.nan
        ratio = (a[valid] - b[valid]) / denom[valid]
        return float(3.0 * np.mean(ratio ** 2))
    return _per_neuron_stat(values, offsets, _lv)


def fano_factor(times: np.ndarray, ids: np.ndarray, num_neurons: int,
                duration_ms: float, bin_ms: float = 100.0) -> np.ndarray:
    """ニューロンごとの Fano factor (窓内スパイク数の分散/平均)。ポアソンで 1。"""
    times = np.asarray(times, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    out = np.full(num_neurons, np.nan, dtype=np.float64)
    if duration_ms <= 0.0 or bin_ms <= 0.0:
        return out
    num_bins = max(1, int(np.ceil(duration_ms / bin_ms)))
    if num_bins < 2:
        return out

    bin_index = np.minimum((times / bin_ms).astype(np.int64), num_bins - 1)
    counts = np.zeros((num_neurons, num_bins), dtype=np.float64)
    keep = (ids >= 0) & (ids < num_neurons)
    np.add.at(counts, (ids[keep], bin_index[keep]), 1.0)

    mean = counts.mean(axis=1)
    var = counts.var(axis=1)
    active = mean > 0.0
    out[active] = var[active] / mean[active]
    return out


def isi_metrics(times: np.ndarray, ids: np.ndarray, num_neurons: int,
                duration_ms: float, bin_ms: float = 100.0) -> dict[str, float]:
    """1 記録窓ぶんの ISI 指標を平坦な dict で返す (metrics.csv の 1 行分)。

    平均・中央値は**間隔が定義できたニューロンだけ**で取る。スパイクが 2 発以下の
    ニューロンを 0 として混ぜると、活動が落ちたときに CV が下がったように見えてしまう
    (「規則的になった」と「黙った」が区別できなくなる)。黙ったことは
    `silent_fraction` が別途報告する。
    """
    times = np.asarray(times, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    values, offsets = isi_per_neuron(times, ids, num_neurons)
    cv = cv_isi(values, offsets)
    lv = local_variation(values, offsets)
    fano = fano_factor(times, ids, num_neurons, duration_ms, bin_ms)

    spike_counts = np.bincount(ids[(ids >= 0) & (ids < num_neurons)], minlength=num_neurons)
    silent = spike_counts == 0

    def _nanmean(a):
        return float(np.nanmean(a)) if np.any(np.isfinite(a)) else float("nan")

    def _nanmedian(a):
        return float(np.nanmedian(a)) if np.any(np.isfinite(a)) else float("nan")

    return {
        "isi_mean_ms": float(np.mean(values)) if values.size else float("nan"),
        "isi_median_ms": float(np.median(values)) if values.size else float("nan"),
        "cv_isi_mean": _nanmean(cv),
        "cv_isi_median": _nanmedian(cv),
        "lv_mean": _nanmean(lv),
        "lv_median": _nanmedian(lv),
        "fano_mean": _nanmean(fano),
        "silent_fraction": float(np.mean(silent)),
        "num_neurons_with_isi": int(np.count_nonzero(np.isfinite(cv))),
    }
