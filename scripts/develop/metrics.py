"""1 記録窓ぶんの指標を 1 行の dict にする。

**本番の run も `replot.py` の再解析も、必ずこの 1 つの関数を通る。** 以前は
`develop.py` の中に本番用と再解析用の 2 実装があり、再解析側にだけ E/I 列と重みブロック列が
無かった。そのせいで再解析は `metrics.csv` を上書きできず別名に逃げるしかなく、fig2c が
どちらを読むかを呼び出し側が明示しないと**例外を出さずに古い値を描く**状態になっていた。
列が一致したことで、再解析は素直に上書きできるようになった。

「何を測るか」はこの実験が決め、「どう測るか」は `src/utils/analysis/` に任せる。
ここにあるのは呼び出しと列名の組み立てだけ。

**入力はすべて必須。** スパイクも layout も重みも揃っていなければ行は作れない、という
一本の規則にしてある。揃わない状況 (`connectivity.npz` の無い古い run など) は、列を
減らして続行するのではなく、読み込む側で落ちること —— 列の欠けた CSV を作ると
「測ったが値が無い」と「そもそも測っていない」が区別できなくなり、あとで図と突き合わせた
ときに原因が追えなくなる。
"""
from __future__ import annotations

import numpy as np

from src.utils.analysis.avalanche import split_avalanches
from src.utils.analysis.criticality import (bimodality_d, burstiness_index,
                                            criticality_index_delta_cr)
from src.utils.analysis.powerlaw import log_likelihood_ratio_power_vs_exponential
from src.utils.analysis.spikes import diagnose_activity, firing_rates, spike_group_metrics
from src.utils.analysis.weights import block_values, weight_block_metrics


def resolve_avalanche_smax(config, override: int | None = None) -> int:
    """べき乗フィット / ΔCr の上限サイズを決める。既定は **システムサイズ N**。

    論文 (Ikeda-Akita-Takahashi 2023) の [1, 100] は N=100 のネットワークの話で、100 は
    定数ではなく **系のサイズそのもの**。Beggs-Plenz 系の慣例どおり、べき乗スケーリング領域の
    カットオフは電極数 / ニューロン数で決まる。よって既定を N にすれば N=100 の config の
    結果は一切変わらないまま、他の N へそのまま外挿できる。

    smax は「データの切り取り」ではなく **モデルの正規化台** であることに注意
    (p(s) = s^-α / Σ_{k=1}^{smax} k^-α)。観測サイズが smax を超えなくても、smax を変えれば
    α は動く。「どうせ 100 を超えないから 100 のままで安全」ではない。
    """
    if override is not None:
        if override < 2:
            raise ValueError(f"avalanche smax は 2 以上である必要があります (got {override})。")
        return int(override)
    return int(config.simulation.N)


def max_plasticity_weight(config) -> float:
    """重み飽和率 (`weight_at_max_fraction`) の分母。可塑性の Wmax の最大値。"""
    wmax_values = []
    for syn_cfg in config.synapses.values():
        wmax = getattr(syn_cfg.plasticity, "Wmax", None)
        if wmax is not None:
            wmax_values.append(float(wmax))
    return max(wmax_values) if wmax_values else 1.0


def spike_columns(local_times, ids, total_neurons: int, record_window_ms: float,
                  smax: int) -> dict:
    """スパイク列だけから決まる指標。`spikes_*h.npz` があれば後からでも再計算できる。

    `local_times` は**記録窓の先頭を 0 とするローカル時刻**。絶対時刻を渡すと
    アバランチ分割は同じでも burstiness のビン割りがずれる。
    """
    local_times = np.asarray(local_times, dtype=np.float64)
    ids = np.asarray(ids)
    avalanche = split_avalanches(local_times)
    rates = firing_rates(ids, total_neurons, record_window_ms)
    return {
        "num_spikes": int(local_times.size),
        "mean_rate_hz": float(np.mean(rates)),
        "avalanche_threshold_ms": avalanche.threshold_ms,
        "num_avalanches": int(avalanche.sizes.size),
        "avalanche_smax": int(smax),
        "llr": log_likelihood_ratio_power_vs_exponential(avalanche.sizes),
        "delta_cr": criticality_index_delta_cr(avalanche.sizes, smax=smax),
        "burstiness_index": burstiness_index(local_times, record_window_ms),
        "bimodality_d": bimodality_d(avalanche.sizes),
    }


def group_columns(ids, layout, record_window_ms: float) -> dict:
    """E/I 別の発火指標。layout があれば再解析でも計算できる。"""
    polarity = layout.ids_by("polarity")
    return spike_group_metrics(np.asarray(ids), polarity.get("excitatory"),
                               polarity.get("inhibitory"), record_window_ms)


def weight_columns(weights, row, col, layout, wmax: float) -> dict:
    """重みブロック (EE / EI / IE / II) の指標。

    **スパイクからは再計算できない列。** `weights_*h.npz` と `connectivity.npz` の両方が
    要る。O(nnz) — COO は実結合しか持たないので、結合の無い箇所の 0 が統計に混ざらない。
    """
    blocks = block_values(np.asarray(weights), np.asarray(row), np.asarray(col), layout)
    return weight_block_metrics(blocks, wmax=wmax)


def build_row(
    hour: float,
    local_times,
    ids,
    *,
    total_neurons: int,
    record_window_ms: float,
    smax: int,
    layout,
    weights,
    row,
    col,
    wmax: float,
) -> dict:
    """`metrics.csv` の 1 行を作る。

    Args:
        hour: 記録時刻 [h]。行の識別子であり、fig2c / fig2d の x 軸。
        local_times: 記録窓の先頭を 0 とするスパイク時刻 [ms]。絶対時刻を渡すと
            アバランチ分割は同じでも burstiness のビン割りがずれる。
        ids: スパイクのグローバル ID。
        layout: E/I の判定に使う `NetworkLayout`。
        weights / row / col: COO の重み。`connectivity.npz` と `weights_{h}h.npz` から。
        wmax: 重み飽和率の分母 (`max_plasticity_weight(config)`)。
    """
    out: dict = {"hour": float(hour)}
    out.update(spike_columns(local_times, ids, total_neurons, record_window_ms, smax))
    out.update(group_columns(ids, layout, record_window_ms))
    out.update(weight_columns(weights, row, col, layout, wmax))
    out.update(diagnose_activity(
        mean_rate_hz=out["mean_rate_hz"],
        weight_at_max_fraction=out["weight_at_max_fraction"],
    ))
    return out
