"""損傷 run 1 本について、各ニューロンが入力から受ける駆動の収支 (切断で何を失い、回復で何を得たか) を測る。

**問い。** 切断でニューロン b の発火率がどれだけ下がり、20 h までにどれだけ戻ったかを、b に入る
結合 k→b の駆動の変化で説明する。低下は「切られた入力を直接失った分」と「残った入力の上流 k が
落ちた分」に、回復は「重みの変化 (強化・弱化) で得た / 失った分」と「上流 k が戻った分」に分ける。

## 駆動

結合 k→b の窓 t での駆動 g_kb(t) = w_kb(t) · u_k(t) [nS] (時間平均コンダクタンス)。
w_kb(t) は**窓の先頭**の重み、u_k(t) は k の重み 1 あたりの駆動 (`unit_drive()`)。式は `drive.py` と
同じで、b ごとに足すと `drive_columns()` の値になる。時点は切断前 (`pre`、親 run から引き継いだ
重みと親の記録窓)、0 h (切断と settle の後の最初の probe)、20 h。

## 駆動の変化の列 (結合 1 本ごと。失った量は正)

| 列 | 切られた結合 | 残った結合 |
|---|---|---|
| `lost_direct` (切られて失った駆動) | g(pre) | 0 |
| `lost_upstream` (切られずに失った駆動) | 0 | g(pre) − g(0h) |
| `gain` (0 h から得た駆動) | 0 | g(20h) − g(0h) |
| `drive_gain` (k の重み 1 あたりの駆動の変化) | 空 | u_k(20h) − u_k(0h) |

`lost_upstream` は k の発火の変化と settle (`task.settle_ms`、可塑性は動いている) 中の重みの変化の両方を
含む。`gain` は重みの変化と k の発火の変化の両方を含み、`gain` = (w(20h) − w(0h)) u_k(20h) +
w(0h) · `drive_gain` と分かれる。b に入る駆動は G_b(0h) − G_b(pre) = −Σ`lost_direct` − Σ`lost_upstream`、
G_b(20h) − G_b(0h) = Σ`gain`。

## 迂回経路上の位置 (`cls`、残った E→E のみ)

切られた強い E→E a→b を基準に、切断後に残った E→E でたどった位置 (`detours.detour_positions()`)。
`rev` (逆向き b→a)、`len{n}_hop{i}` (長さ n ≤ 3 の迂回経路の a から数えて i 段目)、`other`。
当てはまる位置はすべて `;` でつないで記録する (例 `len2_hop2;len3_hop3`)。

## 表は 2 つ

- `input_budget()`: 結合 k→b 1 本 = 1 行 (切った結合も含む)。post → pre の順に並べる
- `neuron_budget()`: ニューロン b 1 個 = 1 行。発火率・低下・回復、入力の本数 (`n_in_*` は切断前、
  `n_kept_*` は切られずに残った分) と、駆動の列を b ごとに E 発の入力 (`_e`) と I 発の入力 (`_i`) に
  限って足したもの

低下 = (r_pre − r_0h) / r_pre、回復 = (r_20h − r_0h) / r_pre (r は窓の発火率)。r_pre = 0 なら空。
強化 / 弱化 (`change`) は 0 h → 20 h の重みで `STRONG` を下から上 / 上から下へまたいだ残った結合。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.analysis.weights import excitatory_flags
from src.utils.runview import MissingData

from scripts.lesion.analysis.detours import STRONG, detour_positions
from scripts.lesion.analysis.drive import (conductance_per_release, released_per_neuron,
                                           source_params)
from scripts.lesion.analysis.flip_rule import _pre_weights

# 回復を測る probe の時刻 [h] (切断からの経過)。
TARGET_HOUR = 20.0
# 駆動の変化の列。b の表では E 発と I 発の入力に分けて足す。
TERMS = ("lost_direct", "lost_upstream", "gain", "drive_gain")



def _window_at(series, hour: float):
    match = [w for w in series.windows[1:] if abs(float(w.hour) - hour) < 1e-6]
    if len(match) != 1:
        raise MissingData("input budget", f"切断から {hour:g} h の probe が {len(match)} 個あります")
    return match[0]


def unit_drive(window, params, dt_ms: float, num_neurons: int) -> np.ndarray:
    """窓の中で各ニューロンが出す、重み 1 あたりの駆動 [nS]。長さ N。"""
    spikes = window.spikes()
    released = released_per_neuron(spikes.times, spikes.ids, params, num_neurons)
    return released * conductance_per_release(params, dt_ms) * 1000.0 / float(window.record_window_ms)


def firing_rates(window, num_neurons: int) -> np.ndarray:
    """窓の中の各ニューロンの発火率 [Hz]。長さ N。"""
    ids = np.asarray(window.spikes().ids, dtype=np.int64)
    return np.bincount(ids, minlength=num_neurons) / (float(window.record_window_ms) / 1000.0)


def change_ratios(r_pre, r_0h, r_20h) -> tuple[np.ndarray, np.ndarray]:
    """発火率から (低下, 回復) を返す。r_pre = 0 の要素は NaN。"""
    r_pre, r_0h, r_20h = (np.asarray(r, dtype=np.float64) for r in (r_pre, r_0h, r_20h))
    with np.errstate(divide="ignore", invalid="ignore"):
        drop = np.where(r_pre > 0, (r_pre - r_0h) / r_pre, np.nan)
        recovery = np.where(r_pre > 0, (r_20h - r_0h) / r_pre, np.nan)
    return drop, recovery


def crossings(cut, w_0h, w_20h) -> tuple[np.ndarray, np.ndarray]:
    """0 h → 20 h で `STRONG` を下から上 (強化) / 上から下 (弱化) へまたいだ残った結合のマスク。"""
    kept = ~np.asarray(cut, dtype=bool)
    with np.errstate(invalid="ignore"):
        return (kept & (w_0h <= STRONG) & (w_20h > STRONG),
                kept & (w_0h > STRONG) & (w_20h <= STRONG))


def budget_terms(cut, g_pre, g_0h, g_20h, u_0h, u_20h) -> dict[str, np.ndarray]:
    """結合ごとの駆動の変化の列 (`TERMS`) を返す。どれも結合の並びの配列。

    `g_*` は各窓の駆動 (切られた結合の `g_0h` / `g_20h` は使わない)、`u_*` は pre の重み 1 あたりの駆動。
    """
    cut = np.asarray(cut, dtype=bool)
    g_pre, g_0h, g_20h, u_0h, u_20h = (np.asarray(v, dtype=np.float64)
                                       for v in (g_pre, g_0h, g_20h, u_0h, u_20h))
    return {
        "lost_direct": np.where(cut, g_pre, 0.0),
        "lost_upstream": np.where(cut, 0.0, g_pre - g_0h),
        "gain": np.where(cut, 0.0, g_20h - g_0h),
        "drive_gain": np.where(cut, np.nan, u_20h - u_0h),
    }


def input_budget(series) -> pd.DataFrame:
    """損傷 run 1 本の、切断前の結合 k→b 1 本 = 1 行の駆動の収支を返す。

    列: `pre` / `post` / `pre_exc` / `post_exc` / `cut`、重み `w_pre` / `w_0h` / `w_20h`、
    `change` (`strengthened` / `weakened` / 空)、駆動 `g_pre` / `g_0h` / `g_20h`、
    `pre_drop` / `pre_recovery` (k 自身)、`cls` (残った E→E のみ。ほかは空)、`TERMS` の各列。
    post → pre の順に並べる。
    """
    n = int(series.layout.total_neurons)
    exc = excitatory_flags(series.layout, n)
    base, at_0h, at_20h = series.windows[0], _window_at(series, 0.0), _window_at(series, TARGET_HOUR)

    pre_index, w_pre, row_kept, col_kept, _ = _pre_weights(series)
    wiring = series.wiring_pre()
    row = np.asarray(wiring.row, dtype=np.int64)
    col = np.asarray(wiring.col, dtype=np.int64)
    kept_at = np.array([pre_index[p] for p in zip(row_kept.tolist(), col_kept.tolist())],
                       dtype=np.int64)
    cut = np.ones(row.size, dtype=bool)
    cut[kept_at] = False

    def on_pre_order(window):
        out = np.full(row.size, np.nan)
        out[kept_at] = np.asarray(window.weights(), dtype=np.float64)
        return out

    w_0h, w_20h = on_pre_order(at_0h), on_pre_order(at_20h)
    params = source_params(series.config, series.layout)
    dt = float(series.config.simulation.dt)
    u_pre, u_0h, u_20h = (unit_drive(w, params, dt, n)[row] for w in (base, at_0h, at_20h))
    g_pre = w_pre * u_pre
    g_0h, g_20h = np.where(cut, 0.0, w_0h * u_0h), np.where(cut, 0.0, w_20h * u_20h)

    rates = [firing_rates(w, n) for w in (base, at_0h, at_20h)]
    drop, recovery = change_ratios(*rates)

    ee_cut = [(int(a), int(b)) for a, b, c, w in zip(row, col, cut, w_pre)
              if c and exc[a] and exc[b] and w > STRONG]
    ee_kept = np.flatnonzero(~cut & exc[row] & exc[col])
    cls = np.full(row.size, "", dtype=object)
    cls[ee_kept] = detour_positions([(int(row[k]), int(col[k])) for k in ee_kept], ee_cut)

    up, down = crossings(cut, w_0h, w_20h)
    change = np.full(row.size, "", dtype=object)
    change[up], change[down] = "strengthened", "weakened"

    frame = pd.DataFrame(dict(
        pre=row, post=col, pre_exc=exc[row], post_exc=exc[col], cut=cut,
        w_pre=w_pre, w_0h=w_0h, w_20h=w_20h, change=change,
        g_pre=g_pre, g_0h=g_0h, g_20h=g_20h,
        pre_drop=drop[row], pre_recovery=recovery[row], cls=cls,
        **budget_terms(cut, g_pre, g_0h, g_20h, u_0h, u_20h)))
    frame = frame.sort_values(["post", "pre"], kind="stable").reset_index(drop=True)
    frame.insert(0, "seed", series.config.simulation.seed)
    return frame


def neuron_budget_from_inputs(inputs: pd.DataFrame, rates, exc) -> pd.DataFrame:
    """`input_budget()` の表を後ニューロンごとに足し、ニューロン 1 個 = 1 行の表を返す。

    `rates` は (r_pre, r_0h, r_20h) の 3 本 (長さ N)、`exc` は長さ N の極性。
    """
    n = len(exc)
    exc = np.asarray(exc, dtype=bool)
    r_pre, r_0h, r_20h = (np.asarray(r, dtype=np.float64) for r in rates)
    drop, recovery = change_ratios(r_pre, r_0h, r_20h)
    post = inputs["post"].to_numpy()
    from_e = inputs["pre_exc"].to_numpy(dtype=bool)
    cut = inputs["cut"].to_numpy(dtype=bool)
    strong = inputs["w_pre"].to_numpy() > STRONG

    def per_post(values, mask):
        return np.bincount(post[mask], weights=np.asarray(values, dtype=np.float64)[mask],
                           minlength=n)

    def count(mask):
        return np.bincount(post[mask], minlength=n)

    out = dict(neuron=np.arange(n), exc=exc, rate_pre=r_pre, rate_0h=r_0h, rate_20h=r_20h,
               drop=drop, recovery=recovery,
               n_cut_strong_e=count(cut & from_e & strong),
               n_cut_other=count(cut & ~(from_e & strong)))
    for suffix, origin in (("e", from_e), ("i", ~from_e)):
        out[f"n_in_{suffix}"] = count(origin)
        out[f"n_kept_{suffix}"] = count(origin & ~cut)
        for column in ("g_pre", "g_0h", "g_20h"):
            out[f"drive_{column[2:]}_{suffix}"] = per_post(inputs[column], origin)
        for change in ("strengthened", "weakened"):
            out[f"n_{change}_{suffix}"] = count(origin & (inputs["change"] == change).to_numpy())
        for term in TERMS:
            out[f"{term}_{suffix}"] = per_post(np.nan_to_num(inputs[term].to_numpy(dtype=np.float64)),
                                               origin)
    frame = pd.DataFrame(out)
    frame.insert(0, "seed", inputs["seed"].iloc[0] if len(inputs) else "")
    return frame


def neuron_budget(series) -> pd.DataFrame:
    """損傷 run 1 本の、ニューロン 1 個 = 1 行の駆動の収支を返す (列はモジュール docstring)。"""
    n = int(series.layout.total_neurons)
    windows = (series.windows[0], _window_at(series, 0.0), _window_at(series, TARGET_HOUR))
    return neuron_budget_from_inputs(input_budget(series), [firing_rates(w, n) for w in windows],
                                     excitatory_flags(series.layout, n))


def _write(frame: pd.DataFrame, out_path) -> None:
    from scripts.lesion.store.formats import write_table

    write_table(frame.round(6).astype(object).where(frame.notna(), "").to_dict("records"),
                out_path)


def write_input_budget(series, out_path) -> None:
    """`input_budget()` を CSV へ書く。形は `(view, out_path)` (view = run 全体の `Series`)。"""
    _write(input_budget(series), out_path)


def write_neuron_budget(series, out_path) -> None:
    """`neuron_budget()` を CSV へ書く。形は `(view, out_path)` (view = run 全体の `Series`)。"""
    _write(neuron_budget(series), out_path)
