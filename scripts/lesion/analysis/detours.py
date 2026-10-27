"""切断された E→E シナプスの迂回経路と、切断後にそれがどう組み変わったかを測る。

**問い。** 結合 a→b を切ったとき、a から b へ回り道で届く経路 (順方向の迂回 a⇝b) と、
逆向きの経路 (b⇝a) はどれだけあり、切断後に強まったのか。回復の一部は、失った直接の
結合の代わりに迂回経路が強まること (伝達経路の作り直し) で起きる、という仮説を確かめる。

## 定義

- **迂回経路**: 切断後に残った E→E 結合だけでたどる、長さ 2〜`MAX_LENGTH` の単純な経路。
  途中に a, b を通らない。重みで足切りはしない (弱い経路も数える)
- **経路の重み**: 経路上の最小の重み (ボトルネック)。`STRONG` を超えれば「強い経路」
- **強化 / 弱化**: 切断前 → 最後の probe で、`STRONG` を下から上 / 上から下へまたいだ
- **対象**: 切断した E→E (`kind = cut`) と、比べるための切らなかった E→E (`kind = kept`)。
  どちらも切断前の重みで `strong` (> `STRONG`) と `zero` (≤ `STRONG`) に分ける
- **post 先行率**: 結合 pre→post について、±`LEAD_WINDOW_MS` 以内のスパイク対のうち
  post が先に撃った対の割合 (スパイク時刻で測る。遅延は足さない)。0.5 より大きいほど
  「post が先」で、加算型 STDP では LTD 側に押される

**時点は 3 つ**: 切断前 (`pre` = ベースライン窓。親 run の記録)、切断直後 (`p0` = 0 h の probe)、
最後の probe (`end`)。

E→I や I 発の結合は対象にしない。抑制性ニューロンを経由する経路は興奮を中継しないので、
「迂回」の意味が変わるため。

**条件の列は `spec` と `frozen` の 2 つ** (`condition_of()`)。どちらも `lesion.json` から作り、
run ディレクトリ名は使わない。表はこの 2 列で seed をまたいでプールする。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.analysis.weights import excitatory_flags
from src.utils.runview import MissingData

# 「強い」とみなす重み。加算型・ハード境界の STDP では重みが 0 と 1 に二極化するので、
# 中点で切れば境界の取り方にほとんど依らない。
STRONG = 0.5
# 迂回経路の最大の長さ (シナプス数)。3 段を超えると確率発火で伝達がほぼ届かない。
MAX_LENGTH = 3
# post 先行率を数える時間窓 [ms]。e-STDP の時定数 (τ_E = 20 ms) に合わせる。
LEAD_WINDOW_MS = 20.0

# 条件を表す列。どの表もこの 2 列と `seed` を持つ。
CONDITION = ("spec", "frozen")
# 対照 (切らずに走り続けた run) の行の `spec`。
CONTROL_SPEC = "control (no cut)"


def condition_of(manifest: dict) -> dict[str, str]:
    """`lesion.json` から条件の 2 列を作る。

    `spec` は切断 spec を `;` で連結したもの (`runs.csv` の `spec` と同じ)、`frozen` は止めた
    可塑性の群を `-` で連結したもの。`frozen_plasticity` を持たない古い run と、何も止めて
    いない run はどちらも `""`。
    """
    return {"spec": ";".join(manifest["cut_specs"]),
            "frozen": "-".join(sorted(manifest.get("frozen_plasticity") or []))}


@dataclass(frozen=True)
class RunDetours:
    """run 1 本の測定結果。どの表にも条件 (`CONDITION`) と `seed` の列が付く。

    targets   対象シナプス 1 本 = 1 行 (逆向き結合の重みと、迂回経路の本数)
    paths     迂回経路 1 本 = 1 行 (ボトルネックの 3 時点と、強化された辺の位置)
    last_hop  順方向の迂回経路の最後の段 (k→b) 1 本 = 1 行 (post 先行率の 3 時点)
    turnover  切らなかった E→E の強化 / 弱化の本数 (1 行)
    """
    targets: pd.DataFrame
    paths: pd.DataFrame
    last_hop: pd.DataFrame
    turnover: pd.DataFrame


def spike_trains(spikes, num_neurons: int) -> list[np.ndarray]:
    """スパイク列をニューロンごとの時刻配列 (昇順) に分ける。"""
    ids = np.asarray(spikes.ids, dtype=np.int64)
    times = np.asarray(spikes.times, dtype=np.float64)
    order = np.lexsort((times, ids))
    ids, times = ids[order], times[order]
    bounds = np.searchsorted(ids, np.arange(num_neurons + 1))
    return [times[bounds[k]:bounds[k + 1]] for k in range(num_neurons)]


def lead_fraction(trains, pre, post, window_ms: float = LEAD_WINDOW_MS) -> np.ndarray:
    """結合 pre→post ごとに、±`window_ms` 以内のスパイク対のうち post が先に撃った割合。

    対が 1 つも無い結合は NaN。同時刻の対は「post が先」に数えない。
    """
    out = np.full(len(pre), np.nan)
    for s, (i, j) in enumerate(zip(np.asarray(pre), np.asarray(post))):
        a, b = trains[i], trains[j]
        if a.size == 0 or b.size == 0:
            continue
        lo = np.searchsorted(a, b - window_ms, side="left")
        hi = np.searchsorted(a, b + window_ms, side="right")
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            continue
        # b の各スパイクについて、窓内で a がそれより後ろにあるもの = post が先。
        after = hi - np.searchsorted(a, b, side="right")
        out[s] = float(np.clip(after, 0, None).sum()) / total
    return out


def enumerate_paths(adjacency, edge_index, a: int, b: int, max_length: int = MAX_LENGTH):
    """a から b への長さ 2〜`max_length` の単純な経路を、辺番号のタプルで列挙する。

    `adjacency[i]` は i から出る `(行き先, 辺番号)` の並び、`edge_index[(i, j)]` は辺番号。
    """
    found = []

    def walk(node, visited, edges):
        depth = len(edges)
        last = edge_index.get((node, b))
        if depth >= 1 and last is not None:
            found.append(tuple(edges) + (last,))
        if depth + 1 >= max_length:
            return
        for nxt, edge in adjacency[node]:
            if nxt in visited or nxt == b:
                continue
            walk(nxt, visited | {nxt}, edges + [edge])

    walk(a, {a, b}, [])
    return found


def _flip_positions(edges, before, after) -> tuple[int, int, int]:
    """経路のうち STRONG を下から上へまたいだ辺を、位置 (先頭 / 中間 / 末尾) ごとに数える。"""
    first = middle = last = 0
    for position, edge in enumerate(edges):
        if before[edge] <= STRONG < after[edge]:
            if position == 0:
                first += 1
            elif position == len(edges) - 1:
                last += 1
            else:
                middle += 1
    return first, middle, last


def run_detours(series) -> RunDetours:
    """損傷 run 1 本の迂回経路を測る。

    `series` は損傷 run の `Series`。ベースライン窓 (切断前の重みとスパイク) と、切断直後・
    最後の probe が書かれている必要がある。
    """
    condition = condition_of(series.manifest())
    n = int(series.layout.total_neurons)
    exc = excitatory_flags(series.layout, n)
    seed = series.config.simulation.seed
    windows = series.windows
    if len(windows) < 3:
        raise ValueError(f"{series.run_dir}: 切断前・0 h・最後の probe の 3 窓が要ります "
                         f"({len(windows)} 窓しかありません)")
    baseline, first, final = windows[0], windows[1], windows[-1]

    pre = series.wiring_pre()
    pre_index = {(i, j): k for k, (i, j) in
                 enumerate(zip(np.asarray(pre.row).tolist(), np.asarray(pre.col).tolist()))}
    w_pre_all = np.asarray(baseline.weights(), dtype=np.float64)

    post = series.wiring()
    row, col = np.asarray(post.row, dtype=np.int64), np.asarray(post.col, dtype=np.int64)
    edge_index = {(i, j): k for k, (i, j) in enumerate(zip(row.tolist(), col.tolist()))}
    w = {"pre": np.array([w_pre_all[pre_index[(i, j)]] for i, j in zip(row.tolist(), col.tolist())]),
         "p0": np.asarray(first.weights(), dtype=np.float64),
         "end": np.asarray(final.weights(), dtype=np.float64)}

    ee = exc[row] & exc[col]
    ee_index = {key: k for key, k in edge_index.items() if ee[k]}
    adjacency = [[] for _ in range(n)]
    for k in np.flatnonzero(ee):
        adjacency[row[k]].append((int(col[k]), int(k)))

    cut = series.cut()
    cut_row, cut_col = np.asarray(cut["row"], dtype=np.int64), np.asarray(cut["col"], dtype=np.int64)
    cut_pairs = set(zip(cut_row.tolist(), cut_col.tolist()))
    targets = [("cut", int(a), int(b), float(w_pre_all[pre_index[(a, b)]]))
               for a, b in zip(cut_row.tolist(), cut_col.tolist()) if exc[a] and exc[b]]
    targets += [("kept", int(row[k]), int(col[k]), float(w["pre"][k])) for k in np.flatnonzero(ee)]

    target_rows, path_rows = [], []
    last_hop_owner = {}   # 辺番号 -> "cut" | "kept" (切った強い結合の迂回を優先)
    for tid, (kind, a, b, w_ab) in enumerate(targets):
        cls = "strong" if w_ab > STRONG else "zero"
        reverse_pre = pre_index.get((b, a))
        reverse_post = edge_index.get((b, a))
        record = dict(target=tid, kind=kind, cls=cls, a=a, b=b, w_pre=w_ab,
                      reverse_exists=reverse_pre is not None,
                      reverse_cut=(b, a) in cut_pairs,
                      reverse_w_pre=w_pre_all[reverse_pre] if reverse_pre is not None else np.nan,
                      reverse_w_end=w["end"][reverse_post] if reverse_post is not None else np.nan)
        for direction, (src, dst) in (("fwd", (a, b)), ("rev", (b, a))):
            found = enumerate_paths(adjacency, ee_index, src, dst)
            for length in range(2, MAX_LENGTH + 1):
                strong = [p for p in found if len(p) == length and w["pre"][list(p)].min() > STRONG]
                record[f"{direction}_strong_L{length}"] = len(strong)
            record[f"{direction}_paths"] = len(found)
            for edges in found:
                e = list(edges)
                first_flip, middle_flip, last_flip = _flip_positions(edges, w["pre"], w["end"])
                path_rows.append(dict(target=tid, dir=direction, length=len(e),
                                      bn_pre=w["pre"][e].min(), bn_p0=w["p0"][e].min(),
                                      bn_end=w["end"][e].min(),
                                      flip_first=first_flip, flip_middle=middle_flip,
                                      flip_last=last_flip))
                if direction == "fwd" and cls == "strong":
                    if kind == "cut" or last_hop_owner.get(e[-1]) != "cut":
                        last_hop_owner[e[-1]] = kind
        target_rows.append(record)

    hop_edges = np.array(sorted(last_hop_owner), dtype=np.int64)
    trains = {name: spike_trains(window.spikes(), n)
              for name, window in (("pre", baseline), ("p0", first), ("end", final))}
    last_hop = pd.DataFrame(dict(
        edge=hop_edges, owner=[last_hop_owner[e] for e in hop_edges.tolist()],
        k=row[hop_edges], b=col[hop_edges],
        w_pre=w["pre"][hop_edges], w_p0=w["p0"][hop_edges], w_end=w["end"][hop_edges],
        **{f"lead_{name}": lead_fraction(trains[name], row[hop_edges], col[hop_edges])
           for name in ("pre", "p0", "end")}))

    kept = ee
    weak, strong = kept & (w["pre"] <= STRONG), kept & (w["pre"] > STRONG)
    turnover = pd.DataFrame([dict(
        weak=int(weak.sum()), weak_to_strong=int((weak & (w["end"] > STRONG)).sum()),
        strong=int(strong.sum()), strong_to_weak=int((strong & (w["end"] <= STRONG)).sum()),
        hours=float(final.hour) - float(first.hour))])

    def tag(frame):
        frame.insert(0, "seed", seed)
        for position, name in enumerate(CONDITION):
            frame.insert(position, name, condition[name])
        return frame

    return RunDetours(targets=tag(pd.DataFrame(target_rows)), paths=tag(pd.DataFrame(path_rows)),
                      last_hop=tag(last_hop), turnover=tag(turnover))


def control_turnover(wiring, w_start, w_end, is_exc, hours: float, seed) -> pd.DataFrame:
    """切らずに走った run で、E→E の強化 / 弱化を `run_detours().turnover` と同じ形で数える。"""
    row, col = np.asarray(wiring.row), np.asarray(wiring.col)
    ee = is_exc[row] & is_exc[col]
    weak, strong = ee & (w_start <= STRONG), ee & (w_start > STRONG)
    return pd.DataFrame([dict(
        spec=CONTROL_SPEC, frozen="", seed=seed,
        weak=int(weak.sum()), weak_to_strong=int((weak & (w_end > STRONG)).sum()),
        strong=int(strong.sum()), strong_to_weak=int((strong & (w_end <= STRONG)).sum()),
        hours=float(hours))])


# ======================================================================================
# 束 (同じ親条件の損傷 run 全部 + 対照) を測る
# ======================================================================================

@dataclass(frozen=True)
class PooledDetours:
    """束全体の測定結果。表はどれも条件 (`CONDITION`) と `seed` の列を持つ。"""
    targets: pd.DataFrame
    paths: pd.DataFrame
    last_hop: pd.DataFrame
    turnover: pd.DataFrame


def _control_turnover(control, cut_hour: float, hours: float) -> pd.DataFrame:
    """対照 run 1 本の E→E の入れ替わりを、切断時刻から `hours` 後までで数える。"""
    start, end = control.state_at(cut_hour), control.state_at(cut_hour + hours)
    layout = control.layout()
    is_exc = excitatory_flags(layout, int(layout.total_neurons))
    return control_turnover(start.wiring, np.asarray(start.weights, dtype=np.float64),
                            np.asarray(end.weights, dtype=np.float64), is_exc, hours,
                            control.config.simulation.seed)


def measure_collection(collection) -> PooledDetours:
    """束の損傷 run を全部測り、対照の E→E の入れ替わりも足して返す。

    対照は「切断時刻から、損傷 run の最後の probe と同じ時間だけ後」までを数える。
    損傷 run どうしで切断時刻か観察時間が違う束は止める (対照の区間が 1 つに決まらない)。
    E→E を 1 本も切っていない束は `MissingData`。
    """
    results, cut_hours = [], set()
    for series in collection.lesion_runs():
        cut_hours.add(float(series.manifest()["restored_from_hour"]))
        result = run_detours(series)
        if (result.targets.kind == "cut").any():
            results.append(result)
    if not results:
        raise MissingData("detours", f"{collection.cond_dir}: E→E を切った損傷 run がありません")

    spans = {round(float(h), 6) for r in results for h in r.turnover.hours}
    if len(cut_hours) != 1 or len(spans) != 1:
        raise ValueError(f"切断時刻 {sorted(cut_hours)} か観察時間 {sorted(spans)} が"
                         " run によって違います")
    cut_hour, span = cut_hours.pop(), spans.pop()
    controls = [_control_turnover(control, cut_hour, span)
                for control in collection.control_runs()]

    def cat(name):
        return pd.concat([getattr(r, name) for r in results], ignore_index=True)

    return PooledDetours(targets=cat("targets"), paths=cat("paths"), last_hop=cat("last_hop"),
                         turnover=pd.concat([cat("turnover"), *controls], ignore_index=True))


# 報告の表を書く関数。形は `(view, out_path)` (view = `PooledDetours`)。
def _write(frame: pd.DataFrame, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(out_path, index=False)


def write_path_summary(pooled: PooledDetours, out_path) -> None:
    _write(path_summary(pooled.targets, pooled.paths), out_path)


def write_reverse_summary(pooled: PooledDetours, out_path) -> None:
    _write(reverse_summary(pooled.targets), out_path)


def write_last_hop_summary(pooled: PooledDetours, out_path) -> None:
    _write(last_hop_summary(pooled.last_hop), out_path)


def write_turnover_summary(pooled: PooledDetours, out_path) -> None:
    _write(turnover_summary(pooled.turnover), out_path)


# ======================================================================================
# 条件 (`CONDITION`) ごとの集計。seed をまたいでプールする。
# ======================================================================================

def path_summary(targets: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    """条件 × 対象の種類 × 向きごとの、迂回経路の本数・重みの分布・切断後の強化。

    本数は「対象 1 本あたり」で、迂回経路を 1 本も持たない対象も分母に入る。
    `flip_*` は強化された経路で STRONG をまたいだ辺の位置の割合 (先頭 = 始点側)。
    """
    targets = targets.assign(group=targets.kind + "-" + targets.cls)
    keys = [*CONDITION, "seed", "target"]
    paths = paths.merge(targets[[*keys, "group"]], on=keys)
    rows = []
    for (*condition, group), tg in targets.groupby([*CONDITION, "group"]):
        in_group = (paths[list(CONDITION)] == condition).all(axis=1) & (paths.group == group)
        for direction in ("fwd", "rev"):
            pg = paths[in_group & (paths.dir == direction)]
            strong_counts = sum(tg[f"{direction}_strong_L{L}"] for L in range(2, MAX_LENGTH + 1))
            up = (pg.bn_pre <= STRONG) & (pg.bn_end > STRONG)
            down = (pg.bn_pre > STRONG) & (pg.bn_end <= STRONG)
            flips = pg.loc[up, ["flip_first", "flip_middle", "flip_last"]].sum()
            total_flips = flips.sum()
            rows.append(dict(
                **dict(zip(CONDITION, condition)), group=group, dir=direction,
                n_seeds=tg.seed.nunique(), n_targets=len(tg), n_paths=len(pg),
                paths_per_target=tg[f"{direction}_paths"].mean(),
                strong_per_target=strong_counts.mean(),
                strong_L2_per_target=tg[f"{direction}_strong_L2"].mean(),
                no_strong_fraction=(strong_counts == 0).mean(),
                bn_below_001=(pg.bn_pre < 0.01).mean(),
                bn_001_to_050=((pg.bn_pre >= 0.01) & (pg.bn_pre <= STRONG)).mean(),
                bn_050_to_099=((pg.bn_pre > STRONG) & (pg.bn_pre < 0.99)).mean(),
                bn_above_099=(pg.bn_pre >= 0.99).mean(),
                up_fraction=up.mean(), down_fraction=down.mean(),
                mean_bn_change=(pg.bn_end - pg.bn_pre).mean(),
                flip_first=flips.flip_first / total_flips if total_flips else np.nan,
                flip_middle=flips.flip_middle / total_flips if total_flips else np.nan,
                flip_last=flips.flip_last / total_flips if total_flips else np.nan))
    return pd.DataFrame(rows)


def reverse_summary(targets: pd.DataFrame) -> pd.DataFrame:
    """強い a→b の逆向き b→a (切断前は 0、自身は切られていない) が最後に強くなったか。

    順方向の強い迂回 a⇝b の短さで分ける。迂回が短いほど「a が先・b が後」の順序が残り、
    b→a は LTD 側に押され続ける、という予想を確かめる表。
    """
    t = targets[(targets.cls == "strong") & targets.reverse_exists & ~targets.reverse_cut
                & (targets.reverse_w_pre < 0.01)].copy()
    t["fwd_detour"] = np.select(
        [t.fwd_strong_L2 > 0, t.fwd_strong_L3 > 0],
        ["strong L2", "strong L3 only"], default=f"none (L<={MAX_LENGTH})")
    out = (t.groupby([*CONDITION, "kind", "fwd_detour"])
           .agg(n=("reverse_w_end", "size"),
                regrown=("reverse_w_end", lambda x: int((x > STRONG).sum())))
           .reset_index())
    out["regrown_fraction"] = out.regrown / out.n
    return out


def last_hop_summary(last_hop: pd.DataFrame) -> pd.DataFrame:
    """順方向の迂回の最後の段 k→b の post 先行率と、弱かった k→b の強化。

    `owner = cut` は切った強い a→b の迂回の末尾、`kept` は切らなかった強い a→b の迂回の末尾。
    「a→b を切ると b が早く撃つ理由を失い、k が先になって k→b が強まる」なら、cut 側の弱い
    k→b で post 先行率が切断直後に下がり、強化が多い。
    """
    rows = []
    for (*condition, owner), g in last_hop.groupby([*CONDITION, "owner"]):
        weak = g[g.w_pre <= STRONG]
        flipped = weak.w_end > STRONG
        shift = weak.lead_p0 - weak.lead_pre
        rows.append(dict(
            **dict(zip(CONDITION, condition)), owner=owner, n_edges=len(g), n_weak=len(weak),
            lead_pre_median=weak.lead_pre.median(), lead_p0_median=weak.lead_p0.median(),
            lead_end_median=weak.lead_end.median(),
            weak_to_strong_fraction=flipped.mean() if len(weak) else np.nan,
            lead_shift_median_flipped=shift[flipped].median(),
            lead_shift_median_not_flipped=shift[~flipped].median(),
            # 0 h の時点でまだ弱いままだった割合。高ければ、先行率のずれ (0 h) は
            # その辺自身の強化の結果ではなく、強化より先に起きていたことになる。
            flipped_still_weak_at_p0=(weak.w_p0[flipped] <= STRONG).mean() if flipped.any() else np.nan,
            strong_lead_pre_median=g[g.w_pre > STRONG].lead_pre.median()))
    return pd.DataFrame(rows)


def turnover_summary(turnover: pd.DataFrame) -> pd.DataFrame:
    """E→E の強化 / 弱化の割合を条件ごとに (対照の行を含む)。"""
    by = list(CONDITION)
    out = turnover.groupby(by)[["weak", "weak_to_strong", "strong",
                                "strong_to_weak"]].sum().reset_index()
    out["weak_to_strong_fraction"] = out.weak_to_strong / out.weak
    out["strong_to_weak_fraction"] = out.strong_to_weak / out.strong
    hours = turnover.groupby(by).hours.agg(["min", "max"]).reset_index()
    return out.merge(hours.rename(columns={"min": "hours_min", "max": "hours_max"}), on=by)
