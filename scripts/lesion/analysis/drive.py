"""シナプス駆動 —— 各ニューロンが E / I から受け取った時間平均コンダクタンス。

損傷で「どれだけ入力を失ったか」と「E と I の釣り合いがどう動いたか」を測る。
発火率や重みだけでは、短期抑圧 (STP) で実際に伝わった量が分からないので、
スパイク列から STP を再現して、**実際に後ニューロンへ届いたコンダクタンス**を積む。

## 何を積むか

シナプス 1 本・プレスパイク 1 発が後ニューロンに与えるコンダクタンスの時間積分:

    x_release · w · g_max  ×  (dt / (1 − exp(−dt/τ_syn)))     [µS·ms]
    └ custom_Akita の addToPost ┘  └ ExpCond の inSyn が減衰しきるまでの和 ┘

これを記録窓の全スパイク・全シナプスで後ニューロンごとに足し、窓の長さで割ると
**時間平均コンダクタンス**になる。E 由来と I 由来を分けて持つ。
列は読みやすさのため nS で出す (config の g_max は µS)。

**電流ではなくコンダクタンス。** 駆動力 (V − E_rev) は膜電位に依存するので掛けない。
E/I 比を電流で読みたいときは、典型的な膜電位で (V − E_exc) / (V − E_inh) を別に掛けること。

## モデルの組み合わせに依存する

上の式は **可塑性 `custom_Akita` × シナプス `ExpCond`** の組でしか正しくない
(STP の漸化式は前者、コンダクタンスの減衰は後者が決める)。だから共有層
(`src/utils/analysis/`) ではなくこの実験に置き、**それ以外の組では計算せずに落とす**。
STP の漸化式そのものはモデル側 (`custom_Akita.replay_released_resource`) が持つ ——
数式の持ち主は 1 か所。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.plasticity.custom_Akita import replay_released_resource
from src.utils.analysis.weights import excitatory_flags

SUPPORTED_PLASTICITY = "custom_Akita"
SUPPORTED_SYNAPSE = "ExpCond"


@dataclass(frozen=True)
class SourceParams:
    """プレニューロンごとの伝達パラメータ。すべて長さ N の配列。"""
    utilization: np.ndarray     # U
    tau_rec: np.ndarray         # [ms]
    g_max: np.ndarray           # [µS]
    tau_syn: np.ndarray         # ExpCond の減衰時定数 [ms]


def source_params(config, layout) -> SourceParams:
    """config の synapse group からプレニューロンごとのパラメータ配列を作る。

    synapse group は source population が出す**全投射**を定義する (target 指定が無い)
    ので、パラメータはプレニューロンだけで決まる。
    """
    n = int(layout.total_neurons)
    arrays = {name: np.full(n, np.nan) for name in ("utilization", "tau_rec", "g_max", "tau_syn")}
    for group_name, group in config.synapses.items():
        plasticity, synapse = group.plasticity, group.synapse
        if plasticity.type != SUPPORTED_PLASTICITY or synapse.type != SUPPORTED_SYNAPSE:
            raise NotImplementedError(
                f"synapses.{group_name} は plasticity={plasticity.type!r} / "
                f"synapse={synapse.type!r}。駆動の計算は {SUPPORTED_PLASTICITY} × "
                f"{SUPPORTED_SYNAPSE} の組にしか対応していません。")
        if getattr(plasticity, "normalize_gmax_by_fan_in", False):
            raise NotImplementedError(
                f"synapses.{group_name} は normalize_gmax_by_fan_in=true。"
                " fan-in 正規化した g_max の駆動計算には対応していません。")
        for population in group.source:
            ids = layout.global_indices(population)
            arrays["utilization"][ids] = float(plasticity.U)
            arrays["tau_rec"][ids] = float(plasticity.tau_rec)
            arrays["g_max"][ids] = float(plasticity.g_max)
            arrays["tau_syn"][ids] = float(synapse.tau)
    missing = np.isnan(arrays["utilization"])
    if missing.any():
        raise ValueError(f"どの synapse group の source にも入っていないニューロンがあります"
                         f" (例: {np.flatnonzero(missing)[:5].tolist()})")
    return SourceParams(**arrays)


def released_per_neuron(times, ids, params: SourceParams, num_neurons: int) -> np.ndarray:
    """記録窓の中で各ニューロンが放出した資源 Σ U·x。長さ N。

    U はニューロンごとの配列のまま渡せるが τ_rec はスカラなので、τ_rec の値ごとに再現する。
    """
    times = np.asarray(times, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    total = np.zeros(int(num_neurons), dtype=np.float64)
    for tau_rec in np.unique(params.tau_rec):
        member = params.tau_rec[ids] == tau_rec
        if not member.any():
            continue
        released = replay_released_resource(times[member], ids[member], num_neurons,
                                             params.utilization, float(tau_rec))
        total += np.bincount(ids[member], weights=released, minlength=int(num_neurons))
    return total


def conductance_per_release(params: SourceParams, dt_ms: float) -> np.ndarray:
    """x_release·w = 1 のとき、1 発が後ニューロンに与えるコンダクタンス積分 [µS·ms]。

    ExpCond は 1 ステップごとに `inSyn` を注入してから `exp(−dt/τ)` で減衰させるので、
    積分は連続時間の g_max·τ ではなく等比級数の和 g_max·dt / (1 − exp(−dt/τ)) になる。
    """
    return params.g_max * float(dt_ms) / (1.0 - np.exp(-float(dt_ms) / params.tau_syn))


def drive_columns(times, ids, row, col, weights, layout, config, window_ms: float) -> dict:
    """`metrics.csv` に入れる駆動の列。1 記録窓ぶん。

    値は**後ニューロン 1 本あたりの時間平均コンダクタンス [nS]** で、後ニューロンの
    極性ごとに平均する。`row` / `col` / `weights` は**その窓の時点の**結合
    (baseline なら切断前、post なら切断後) を渡すこと。

        drive_exc_onto_e_ns   E ニューロンが E から受けた駆動
        drive_inh_onto_e_ns   E ニューロンが I から受けた駆動
        drive_ei_ratio_onto_e 上の 2 つの比 (E / I)
        (onto_i も同様)
    """
    n = int(layout.total_neurons)
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)

    params = source_params(config, layout)
    released = released_per_neuron(times, ids, params, n)
    per_synapse = (released[row] * weights
                   * conductance_per_release(params, float(config.simulation.dt))[row])

    is_exc = excitatory_flags(layout, n)
    from_exc = is_exc[row]
    onto = {
        "exc": np.bincount(col[from_exc], weights=per_synapse[from_exc], minlength=n),
        "inh": np.bincount(col[~from_exc], weights=per_synapse[~from_exc], minlength=n),
    }
    per_ms_to_ns = 1000.0 / float(window_ms)     # [µS·ms] の和 → 時間平均 [nS]
    out = {}
    for suffix, targets in (("e", is_exc), ("i", ~is_exc)):
        exc = float(onto["exc"][targets].mean()) * per_ms_to_ns if targets.any() else float("nan")
        inh = float(onto["inh"][targets].mean()) * per_ms_to_ns if targets.any() else float("nan")
        out[f"drive_exc_onto_{suffix}_ns"] = exc
        out[f"drive_inh_onto_{suffix}_ns"] = inh
        out[f"drive_ei_ratio_onto_{suffix}"] = exc / inh if inh > 0 else float("nan")
    return out
