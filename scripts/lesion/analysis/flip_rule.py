"""切断直後の発火の順序から、切られずに残った E 発の結合が最後に強化 / 弱化されるかを当てられるか測る。

**問い。** e-STDP は加算型・ハード境界なので重みは 0 と 1 に二極化し、行き先はスパイク対の核の和
(ドリフト D) の符号で決まるはず。そうなら、切断直後の窓で測った D の符号が、最後の probe で
重みが `STRONG` をまたぐかを当てる。当たれば「切断で順序がずれ、D の符号が変わった結合だけが
反転する」という規則で、どの結合が作り直されるか (と、その確率) を説明できる。

## 定義

- **対象**: 切られずに残った E 発の結合。切断前 (ベースライン窓) の重みで `weak` (≤ `STRONG`) と
  `strong` に分ける
- **群** (`GROUPS`): `EE` (E→E)、`EI` (E→I)、`EX` (両方)。E→E と E→I は同じ e-STDP 核を持つので、
  順序の観点では区別しない `EX` を、単独の 2 群と並べて出す
- **変化**: `weak` が最後の probe で `STRONG` を超えたら「強化」、`strong` が `STRONG` 以下に
  なったら「弱化」
- **ドリフト D** [w/h]: 窓の中の全スパイク対の e-STDP 核の和 Σ K(t_post − (t_pre + 遅延)) を 1 時間
  あたりにしたもの (`pairing: trace` が足し込む量そのもの。±`DRIFT_WINDOW_MS` より離れた対は捨てる)。
  核のパラメータは run の config の e-STDP の群から読む。可塑性を止めた run でも、止める前の
  学習率で測る (「止めなければどちらへ動いたか」)
- **post 先行率**: `detours.lead_fraction` (±20 ms、遅延は足さない)
- **予測**: `weak` は D > 0 で強化、`strong` は D < 0 で弱化を予測する。AUC は D そのもの
  (`strong` では −D) を得点にした値、`auc_lead` は post 先行率 (`weak` では −先行率) を得点にした値
- **位置関係** (`cls`): 同じ群の切った強い結合 a→b の迂回経路上の位置 (`detours.detour_positions()`)。
  `rev` (逆向き b→a)、`len{n}_hop{i}` (長さ n ≤ 3 の迂回経路の a から数えて i 段目)、`other`。
  1 本が複数に当てはまるときはすべてを `;` でつないで持つ。経路は切断後に残った同じ群の結合で
  たどり、重みで足切りしない。E→I は経路の終点 (b が I) にしかならず、`rev` にもならない
  (E→I a→b の逆向きは I→E で、i-STDP の結合)。E→E の位置関係は E→E の切断だけ、`EI` と `EX` は
  E 発の切断全部に対して決める。**位置ごとの表 (`class_summary()`) では、複数に当てはまる結合を
  その全部の位置に数える** (位置ごとの本数の和は結合の本数より多い)

**時点**: 切断前 (`pre`) と、切断からの時刻が `PROBE_HOURS` の probe (`0h` など)。
条件の列は `detours.condition_of()` と同じ `spec` / `frozen`。表は条件ごとに seed をプールする。

`flip_timing()` は run 1 本の表 (overview が出す)。強化された結合ごとに、D が正になった時刻と
重みが `STRONG` を超えた時刻のどちらが先かを、切断後の probe を全部たどって出す。
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.models.plasticity.custom_Akita import e_stdp_kernel
from src.utils.analysis.weights import excitatory_flags
from src.utils.runview import MissingData

from scripts.lesion.analysis.detours import (CONDITION, OTHER, POSITION_SEP, POSITIONS, STRONG,
                                             condition_of, detour_positions, lead_fraction,
                                             spike_trains)

# 予測に使う probe の時刻 [h] (切断からの経過)。窓は各時刻から `record_window_ms` (10 分)。
PROBE_HOURS = (0.0, 1.0, 2.0)
# ドリフトに入れるスパイク対の時間差の上限 [ms]。e-STDP の τ (20 ms) の 7.5 倍で、核は e^-7.5 に減る。
DRIFT_WINDOW_MS = 150.0
# 群の名前 → post が E か (None は両方)。
GROUPS = {"EE": True, "EI": False, "EX": None}


def e_stdp_params(config) -> dict[str, float]:
    """config から e-STDP の核のパラメータ (`a_e`, `tau_e`, `beta_e`) を読む。

    e-STDP の群 (`plasticity.mode` が `e-stdp` で始まる) がちょうど 1 つでなければ止める。
    """
    found = [group.plasticity for group in config.synapses.values()
             if group.plasticity is not None
             and str(getattr(group.plasticity, "mode", "")).startswith("e-stdp")]
    if len(found) != 1:
        raise ValueError(f"e-STDP の群が {len(found)} 個あります (1 個のときだけ測れます)")
    extra = found[0].model_extra or {}
    return {"a_e": float(extra["A_E"]), "tau_e": float(extra["tau_E"]),
            "beta_e": float(extra["beta_E"])}


def pair_drift(trains, pre, post, delay, kernel, duration_ms: float,
               window_ms: float = DRIFT_WINDOW_MS) -> np.ndarray:
    """結合 pre→post ごとに、全スパイク対の核の和 Σ K(t_post − (t_pre + 遅延)) を 1 時間あたりで返す。

    `trains` は `spike_trains()` の形 (ニューロンごとの昇順の時刻 [ms])。対が無い結合は 0。
    """
    out = np.zeros(len(pre))
    for s, (i, j, d) in enumerate(zip(np.asarray(pre), np.asarray(post), np.asarray(delay))):
        arrival, b = trains[i] + d, trains[j]
        if arrival.size == 0 or b.size == 0:
            continue
        lo = np.searchsorted(arrival, b - window_ms, side="left")
        hi = np.searchsorted(arrival, b + window_ms, side="right")
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            continue
        # post スパイク k の対を、arrival[lo[k]:hi[k]] として 1 本の添字の並びに展開する。
        offsets = np.cumsum(counts) - counts
        index = np.arange(total) + np.repeat(lo - offsets, counts)
        out[s] = float(np.sum(kernel(np.repeat(b, counts) - arrival[index])))
    return out * 3.6e6 / duration_ms


def _probe_windows(windows) -> dict[str, object]:
    """`pre` (ベースライン) と `PROBE_HOURS` の各 probe を、列名の接尾辞 → 窓で返す。"""
    chosen = {"pre": windows[0]}
    for hour in PROBE_HOURS:
        match = [w for w in windows[1:] if abs(float(w.hour) - hour) < 1e-6]
        if len(match) != 1:
            raise MissingData("flip rule", f"切断から {hour:g} h の probe が {len(match)} 個あります")
        chosen[f"{hour:g}h"] = match[0]
    return chosen


def _pre_weights(series):
    """切断前の重みを切断後の結合の並びに揃える。

    (切断前の結合 → 切断前の並びの位置, 切断前の重み, 切断後の row, col, 切断後の並びの切断前の重み)。
    """
    pre = series.wiring_pre()
    pre_index = {(i, j): k for k, (i, j) in
                 enumerate(zip(np.asarray(pre.row).tolist(), np.asarray(pre.col).tolist()))}
    w_pre_all = np.asarray(series.windows[0].weights(), dtype=np.float64)
    post = series.wiring()
    row, col = np.asarray(post.row, dtype=np.int64), np.asarray(post.col, dtype=np.int64)
    w_pre = w_pre_all[[pre_index[p] for p in zip(row.tolist(), col.tolist())]]
    return pre_index, w_pre_all, row, col, w_pre


def run_flips(series) -> pd.DataFrame:
    """損傷 run 1 本の、切られずに残った E 発の結合 1 本 = 1 行の表を返す。

    列: 条件 (`CONDITION`)、`seed`、`pre` / `post` (グローバル ID)、`post_exc`、`w_pre` / `w_end`、
    `cls_ee` (E→E だけ。E→I は空) / `cls_ex`、時点ごとの `drift_<時点>` / `lead_<時点>`。
    """
    condition = condition_of(series.manifest())
    n = int(series.layout.total_neurons)
    exc = excitatory_flags(series.layout, n)
    windows = series.windows
    chosen = _probe_windows(windows)
    final = windows[-1]
    kernel = partial(e_stdp_kernel, **e_stdp_params(series.config))

    pre_index, w_pre_all, row, col, w_pre = _pre_weights(series)
    pairs = list(zip(row.tolist(), col.tolist()))
    w_end = np.asarray(final.weights(), dtype=np.float64)
    delay = np.asarray(series.delays(), dtype=np.float64)

    from_e = exc[row]
    cut = series.cut()
    cut_pairs = [(int(a), int(b)) for a, b in zip(np.asarray(cut["row"]), np.asarray(cut["col"]))
                 if exc[a] and w_pre_all[pre_index[(int(a), int(b))]] > STRONG]
    kept = np.flatnonzero(from_e)
    kept_pairs = [pairs[k] for k in kept]

    ee = kept[exc[col[kept]]]
    cls_ee = dict(zip([pairs[k] for k in ee],
                      detour_positions([pairs[k] for k in ee], [p for p in cut_pairs if exc[p[1]]])))
    cls_ex = detour_positions(kept_pairs, cut_pairs)

    frame = pd.DataFrame(dict(
        pre=row[kept], post=col[kept], post_exc=exc[col[kept]],
        w_pre=w_pre[kept], w_end=w_end[kept],
        cls_ee=[cls_ee.get(p, "") for p in kept_pairs], cls_ex=cls_ex))
    for name, window in chosen.items():
        trains = spike_trains(window.spikes(), n)
        frame[f"drift_{name}"] = pair_drift(trains, row[kept], col[kept], delay[kept], kernel,
                                            float(window.record_window_ms))
        frame[f"lead_{name}"] = lead_fraction(trains, row[kept], col[kept])
    frame.insert(0, "seed", series.config.simulation.seed)
    for position, key in enumerate(CONDITION):
        frame.insert(position, key, condition[key])
    return frame


def flip_timing(series) -> list[dict]:
    """損傷 run 1 本の、強化された E 発の結合 1 本 = 1 行で、D が正になった時刻と重みが `STRONG` を
    超えた時刻を返す。

    強化 = 切断前に `weak` で、最後の probe で `STRONG` を超えた結合 (`run_flips()` と同じ)。
    切断後の probe を全部 (1 時間ごと) たどる。列:

    - `pre` / `post` / `post_exc` / `w_pre` / `w_end` / `drift_0h`: `run_flips()` と同じ意味
    - `drift_hour`: D > 0 になった最初の probe の時刻 [h]。一度もならなければ空
    - `cross_hour`: 重みが `STRONG` を超えた最初の probe の時刻 [h]
    - `first`: `drift` (D が先: `drift_hour < cross_hour`) / `weight` (重みが先か同じ probe、
      または D が一度も正にならない)

    probe の重みは**窓の先頭**の値、D は窓 (10 分) の中で測る。同じ probe で両方が立ったときは、
    重みは前の probe との間に超えていて、その間の窓では D ≤ 0 だったので `weight` に数える。
    """
    n = int(series.layout.total_neurons)
    exc = excitatory_flags(series.layout, n)
    post_cut = series.windows[1:]
    kernel = partial(e_stdp_kernel, **e_stdp_params(series.config))
    _, _, row, col, w_pre = _pre_weights(series)
    w_end = np.asarray(post_cut[-1].weights(), dtype=np.float64)
    chosen = np.flatnonzero(exc[row] & (w_pre <= STRONG) & (w_end > STRONG))
    if chosen.size == 0:
        raise MissingData("flip timing", "強化された E 発の結合がありません")
    delay = np.asarray(series.delays(), dtype=np.float64)[chosen]

    hours = np.array([float(w.hour) for w in post_cut])
    drift = np.empty((len(post_cut), chosen.size))
    crossed = np.empty((len(post_cut), chosen.size), dtype=bool)
    for k, window in enumerate(post_cut):
        trains = spike_trains(window.spikes(), n)
        drift[k] = pair_drift(trains, row[chosen], col[chosen], delay, kernel,
                              float(window.record_window_ms))
        crossed[k] = np.asarray(window.weights(), dtype=np.float64)[chosen] > STRONG

    def first_hour(flags):
        return np.where(flags.any(axis=0), hours[np.argmax(flags, axis=0)], np.nan)

    drift_hour, cross_hour = first_hour(drift > 0), first_hour(crossed)
    return [dict(pre=int(row[s]), post=int(col[s]), post_exc=bool(exc[col[s]]),
                 w_pre=round(float(w_pre[s]), 4), w_end=round(float(w_end[s]), 4),
                 drift_0h=round(float(drift[0, k]), 4),
                 drift_hour="" if np.isnan(drift_hour[k]) else float(drift_hour[k]),
                 cross_hour=float(cross_hour[k]),
                 first="drift" if drift_hour[k] < cross_hour[k] else "weight")
            for k, s in enumerate(chosen)]


def write_flip_timing(series, out_path) -> None:
    """`flip_timing()` を CSV へ書く。形は `(view, out_path)` (view = run 全体の `Series`)。"""
    from scripts.lesion.store.formats import write_table

    write_table(flip_timing(series), out_path)


def measure_collection(collection) -> pd.DataFrame:
    """束の損傷 run を全部測り、`run_flips()` の表をつないで返す。"""
    frames = [run_flips(series) for series in collection.lesion_runs()]
    if not frames:
        raise MissingData("flip rule", f"{collection.cond_dir}: 損傷 run がありません")
    return pd.concat(frames, ignore_index=True)


# ======================================================================================
# 条件 (`CONDITION`) ごとの集計。seed をまたいでプールする。
# ======================================================================================

def auc(score, positive) -> float:
    """`positive` の結合ほど `score` が大きいか (ROC の AUC、同点は 0.5)。どちらかが 0 本なら NaN。"""
    score, positive = np.asarray(score, dtype=np.float64), np.asarray(positive, dtype=bool)
    keep = np.isfinite(score)
    score, positive = score[keep], positive[keep]
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(score)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _group_rows(flips: pd.DataFrame):
    """群ごとに (群の名前, その群の行, 位置関係の列) を返す。"""
    for name, post_exc in GROUPS.items():
        rows = flips if post_exc is None else flips[flips.post_exc == post_exc]
        yield name, rows, rows.cls_ee if name == "EE" else rows.cls_ex


def _windows(flips: pd.DataFrame) -> list[str]:
    return [c.removeprefix("drift_") for c in flips.columns if c.startswith("drift_")]


def _scores(g: pd.DataFrame, window: str, start: str):
    """(変化したか, 変化を予測したか, D の得点, 先行率の得点)。`strong` は符号を反転する。"""
    sign = 1.0 if start == "weak" else -1.0
    changed = (g.w_end > STRONG) if start == "weak" else (g.w_end <= STRONG)
    drift = sign * g[f"drift_{window}"]
    return changed.to_numpy(), (drift > 0).to_numpy(), drift, -sign * g[f"lead_{window}"]


def _rule_row(g: pd.DataFrame, window: str, start: str) -> dict:
    changed, predicted, drift, lead = _scores(g, window, start)
    return dict(
        n=len(g), n_changed=int(changed.sum()),
        changed_fraction=changed.mean() if len(g) else np.nan,
        predicted_fraction=predicted.mean() if len(g) else np.nan,
        # 予測したもののうち変化した割合 / 予測しなかったもののうち変化した割合
        changed_if_predicted=changed[predicted].mean() if predicted.any() else np.nan,
        changed_if_not=changed[~predicted].mean() if (~predicted).any() else np.nan,
        # 変化したもののうち予測できた割合 / 変化しなかったもののうち予測しなかった割合
        sensitivity=predicted[changed].mean() if changed.any() else np.nan,
        specificity=(~predicted[~changed]).mean() if (~changed).any() else np.nan,
        auc_drift=auc(drift, changed), auc_lead=auc(lead, changed),
        drift_median=float(np.median(drift)) * (1.0 if start == "weak" else -1.0) if len(g) else np.nan,
        lead_median=float(np.nanmedian(g[f"lead_{window}"])) if len(g) else np.nan)


def rule_summary(flips: pd.DataFrame) -> pd.DataFrame:
    """条件 × 群 × 切断前の状態 (weak / strong) × 時点ごとの、D の符号による予測の当たり具合。"""
    rows = []
    for group, g_rows, _ in _group_rows(flips):
        for condition, g in g_rows.groupby(list(CONDITION)):
            for start, part in (("weak", g[g.w_pre <= STRONG]), ("strong", g[g.w_pre > STRONG])):
                for window in _windows(flips):
                    rows.append(dict(**dict(zip(CONDITION, condition)), group=group, start=start,
                                     window=window, **_rule_row(part, window, start)))
    return pd.DataFrame(rows)


def class_summary(flips: pd.DataFrame) -> pd.DataFrame:
    """条件 × 群 × 位置関係 × 時点ごとの、弱い結合の強化の確率と D の符号による予測の当たり具合。

    複数の位置に当てはまる結合は、その全部の位置の行に数える。
    """
    rows = []
    for group, g_rows, cls in _group_rows(flips):
        weak = g_rows.w_pre <= STRONG
        g_rows = g_rows[weak].assign(cls=cls[weak].str.split(POSITION_SEP)).explode("cls")
        for (*condition, name), g in g_rows.groupby([*CONDITION, "cls"]):
            for window in _windows(flips):
                rows.append(dict(**dict(zip(CONDITION, condition)), group=group, cls=name,
                                 window=window, **_rule_row(g, window, "weak")))
    out = pd.DataFrame(rows)
    order = {name: k for k, name in enumerate(POSITIONS + (OTHER,))}
    return out.sort_values([*CONDITION, "group", "window", "cls"],
                           key=lambda c: c.map(order) if c.name == "cls" else c,
                           ignore_index=True)


# 報告の表を書く関数。形は `(view, out_path)` (view = `measure_collection()` の表)。
def _write(frame: pd.DataFrame, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(out_path, index=False)


def write_rule_summary(flips: pd.DataFrame, out_path) -> None:
    _write(rule_summary(flips), out_path)


def write_class_summary(flips: pd.DataFrame, out_path) -> None:
    _write(class_summary(flips), out_path)
