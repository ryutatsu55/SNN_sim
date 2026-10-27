"""切断後に片方の STDP を止めた run と止めない run を seed ごとに対にして、回復への各則の寄与を測る。

**問い。** 切断後の活動の回復のうち、どれだけが e-STDP (E 発の結合の重みの変化) から、どれだけが
i-STDP (I 発の結合の重みの変化) から来たか。同じ親・同じ切断で、切断の瞬間から片方の学習率を 0 にした
run (`freeze_plasticity`) と止めない run を比べる。

## 定義

- **回復** rec: `summary.build_row()` の `rec` = (r の `LATE_HOURS` 平均 − r_0h) / r_pre (r はネットワーク
  全体の発火率)。止めた run も止めない run も同じ式
- **止めた則** (`rule`): 止めた群 (`frozen_plasticity`) の `plasticity.mode` の接頭辞 (`e-stdp_…` → `e-STDP`)。
  止めない run は `none`、対照 (切らずに走り続けた run) は `control`
- **寄与** (同じ spec・同じ seed):
  - Δe = rec(i-STDP を止めた run)。e-STDP だけが働いた回復
  - Δi = rec(e-STDP を止めた run)。i-STDP だけが働いた回復
  - Δsum = Δe + Δi。両方止めた run は無いので、rec(none) と比べても加法性は確かめられない (目安)

対象は、止めた run を 1 本でも持つ spec。その spec で `none` / e-STDP / i-STDP の 3 条件がそろう seed だけを使う。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.runview import MissingData

from scripts.lesion.analysis.detours import condition_of
from scripts.lesion.analysis.summary import build_row
from scripts.lesion.store.formats import PHASE_BASELINE, write_table

NONE, CONTROL = "none", "control"
E_RULE, I_RULE = "e-STDP", "i-STDP"
# 3 条件の並び (表の列と図の並びの順)。
RULES = (NONE, E_RULE, I_RULE)
# 推移を描く発火率の列 (`metrics.csv`)。切断前の値で割って `rate` 列にする。
RATE_COLUMN = "mean_rate_hz"


@dataclass(frozen=True)
class FreezeSplit:
    """測定結果。

    runs    1 run = 1 行。`spec` / `seed` / `rule` / `rate_pre` / `rate_0` / `rate_late` / `rec`
    tracks  1 run × 1 probe = 1 行。`spec` / `seed` / `rule` / `hour` (切断からの時刻) と、切断前の値で
            割ったネットワーク全体の発火率 `rate`。対照は `rule="control"` で `spec` は空
    """
    runs: pd.DataFrame
    tracks: pd.DataFrame


def rule_of(config, frozen: list[str]) -> str:
    """止めた群の並びを学習則の名前にする。何も止めていなければ `none`。

    群が 2 つ以上あるとき、または名前の分からない mode なら止める。
    """
    if not frozen:
        return NONE
    if len(frozen) != 1:
        raise ValueError(f"止めた群が {len(frozen)} 個あります (1 個のときだけ比べられます): {frozen}")
    mode = str(config.synapses[frozen[0]].plasticity.mode)
    prefix = mode.split("_")[0]
    rules = {"e-stdp": E_RULE, "i-stdp": I_RULE}
    if prefix not in rules:
        raise ValueError(f"{frozen[0]} の plasticity.mode {mode!r} から学習則が決まりません")
    return rules[prefix]


def _lesion_track(series, spec: str, rule: str) -> pd.DataFrame:
    m = series.metrics()
    pre = m[m.phase == PHASE_BASELINE].iloc[0]
    return pd.DataFrame({"spec": spec, "seed": int(series.config.simulation.seed), "rule": rule,
                         "hour": m.hours_since_cut.to_numpy(dtype=np.float64),
                         "rate": m[RATE_COLUMN].to_numpy(dtype=np.float64) / float(pre[RATE_COLUMN])})


def _control_track(control, cut_hour: float) -> pd.DataFrame:
    m = control.metrics
    at_cut = m[np.isclose(m.hour, cut_hour)]
    if len(at_cut) != 1:
        raise MissingData("control track", f"{control.run_dir}: {cut_hour:g} h の記録がありません")
    return pd.DataFrame({"spec": "", "seed": int(control.config.simulation.seed), "rule": CONTROL,
                         "hour": m.hour.to_numpy(dtype=np.float64) - cut_hour,
                         "rate": m[RATE_COLUMN].to_numpy(dtype=np.float64)
                         / float(at_cut.iloc[0][RATE_COLUMN])})


def complete_seeds(runs: pd.DataFrame) -> pd.DataFrame:
    """(spec, seed) ごとに `RULES` の 3 条件がちょうど 1 本ずつそろう行だけを返す。"""
    complete = runs.groupby(["spec", "seed"]).rule.transform(
        lambda rules: sorted(rules) == sorted(RULES))
    return runs[complete.astype(bool)]


def measure_collection(collection) -> FreezeSplit:
    """束から、止めた run を持つ spec の 3 条件の run と対照を集めて測る。"""
    found = [(series, series.manifest()) for series in collection.lesion_runs()]
    frozen_specs = {condition_of(manifest)["spec"] for _, manifest in found
                    if manifest.get("frozen_plasticity")}
    if not frozen_specs:
        raise MissingData("freeze split", "切断後に可塑性を止めた run がありません")

    rows, tracks = [], []
    for series, manifest in found:
        spec = condition_of(manifest)["spec"]
        if spec not in frozen_specs:
            continue
        rule = rule_of(series.config, manifest.get("frozen_plasticity") or [])
        row = build_row(series)
        rows.append(dict(spec=spec, seed=row["seed"], rule=rule, rate_pre=row["rate_pre"],
                         rate_0=row["rate_0"], rate_late=row["rate_late"], rec=row["rec"],
                         cut_hour=row["cut_hour"], run=row["run"]))
        tracks.append(_lesion_track(series, spec, rule))
    runs = pd.DataFrame(rows)
    complete = complete_seeds(runs)
    dropped = runs.drop(complete.index)[["spec", "seed"]].drop_duplicates()
    for spec, seed in dropped.itertuples(index=False):
        print(f"  対象外 (3 条件がそろわない): spec={spec} seed={seed}")
    if complete.empty:
        raise MissingData("freeze split", "3 条件 (none / e-STDP / i-STDP) がそろう seed がありません")

    tracks = pd.concat(tracks, ignore_index=True).merge(complete[["spec", "seed", "rule"]])
    cut_hours = set(complete.cut_hour)
    if len(cut_hours) != 1:
        raise ValueError(f"切断時刻が run によって違います: {sorted(cut_hours)}")
    cut_hour = cut_hours.pop()
    controls = [_control_track(control, cut_hour) for control in collection.control_runs()]
    tracks = pd.concat([tracks, *controls], ignore_index=True)
    return FreezeSplit(runs=complete.reset_index(drop=True), tracks=tracks.reset_index(drop=True))


def split_table(runs: pd.DataFrame) -> pd.DataFrame:
    """(spec, seed) ごとに 3 条件の rec と寄与 Δe / Δi / Δsum を 1 行にまとめる。"""
    rec = complete_seeds(runs).pivot(index=["spec", "seed"], columns="rule", values="rec")
    out = pd.DataFrame({"rec_none": rec[NONE], "rec_e_frozen": rec[E_RULE],
                        "rec_i_frozen": rec[I_RULE]})
    out["delta_e"] = out.rec_i_frozen
    out["delta_i"] = out.rec_e_frozen
    out["delta_sum"] = out.delta_e + out.delta_i
    return out.reset_index()


def write_freeze_split(measured: FreezeSplit, out_path) -> None:
    """`split_table()` を CSV へ書く。compare の `TABLES` の形 (測る関数の戻り値を受ける)。"""
    frame = split_table(measured.runs)
    write_table(frame.round(6).to_dict("records"), out_path)
