"""損傷 run の束を 1 run = 1 行に要約する (用量反応の表 `runs.csv` / `control.csv`)。

**低下・回復・対照の定義はここにしかない。** 図は表の列を読むだけ。
r = `metrics.csv` の `mean_rate_hz` (ネットワーク全体)、χ = `chi_all`。

    切断数 [%]      cut_pct        = 切った本数 ÷ 切断前の全シナプス数 × 100
    切った重み [%]  cut_weight_pct = lesion.json の cut_profile.weight_lost_fraction × 100
    低下            drop           = (r_pre − r_0) ÷ r_pre
    χ の低下        chi_drop       = 1 − χ_0 ÷ χ_pre
    回復            rec            = (r の LATE_HOURS 平均 − r_0) ÷ r_pre

r_pre は baseline (親 run の切断時刻の窓)、r_0 は切断直後 (0 h) の窓。

**対照も同じ式で測る。** 切らずに走り続けた run で、切断時刻 t0 (損傷 run の
`restored_from_hour`) を 0 h に読み替える。切っていないので r_pre = r_0 = r(t0):

    rec = (r の t0 + LATE_HOURS 平均 − r(t0)) ÷ r(t0)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.lesion.store.records import (CONTROL_TABLE_NAME, PHASE_BASELINE, PHASE_POST,
                                          RUNS_TABLE_NAME, write_table)

# 回復を測る区間 [切断からの h]。両端を含む。
LATE_HOURS = (18.0, 20.0)

# 切断 spec (`<selector>:<k>=<v>,…`) の (selector, pre, post) → 種類。
# pre / post を書かない spec は "any"。**表に無い spec の run は束に入れない。**
KINDS = {
    ("bridge", "any", "any"): "bridge",
    ("random", "any", "any"): "random",
    ("random", "exc", "any"): "random_exc",
}


def kind_of(cut_specs: list[str]) -> str | None:
    """切断 spec の並びから種類を決める。spec が 1 つでないか、表に無ければ None。"""
    if len(cut_specs) != 1:
        return None
    selector, _, args = cut_specs[0].partition(":")
    params = dict(item.split("=", 1) for item in args.split(",") if item)
    return KINDS.get((selector, params.get("pre", "any"), params.get("post", "any")))


def _one(rows, what: str, run_dir):
    if len(rows) != 1:
        raise ValueError(f"{run_dir}: {what} の行が {len(rows)} 個あります (1 個のはず)")
    return rows.iloc[0]


def _late_mean(times, rates, start: float, run_dir) -> float:
    """t ∈ start + LATE_HOURS の平均。**区間の両端が記録に無ければ止める** (途中で止まった run)。"""
    lo, hi = start + LATE_HOURS[0], start + LATE_HOURS[1]
    times, rates = np.asarray(times, dtype=float), np.asarray(rates, dtype=float)
    if not (np.isclose(times, lo).any() and np.isclose(times, hi).any()):
        raise ValueError(f"{run_dir}: 回復を測る {lo:g}–{hi:g} h の記録が揃っていません")
    inside = (times >= lo - 1e-9) & (times <= hi + 1e-9)
    return float(rates[inside].mean())


def build_row(series) -> dict:
    """損傷 run 1 本を 1 行に。種類が `KINDS` に無い run では `kind` が None。"""
    manifest = series.manifest()
    m = series.metrics()
    pre = _one(m[m.phase == PHASE_BASELINE], "baseline", series.run_dir)
    post = m[m.phase == PHASE_POST]
    p0 = _one(post[np.isclose(post.hours_since_cut, 0.0)], "0 h", series.run_dir)
    r_pre, r0 = float(pre.mean_rate_hz), float(p0.mean_rate_hz)
    r_late = _late_mean(post.hours_since_cut, post.mean_rate_hz, 0.0, series.run_dir)
    n_before, n_cut = int(manifest["num_synapses_before"]), int(manifest["num_synapses_cut"])
    return dict(
        run=str(series.run_dir),
        kind=kind_of(manifest["cut_specs"]),
        spec=";".join(manifest["cut_specs"]),
        seed=int(series.config.simulation.seed),
        backend=series.config.simulation.backend,
        cut_hour=float(manifest["restored_from_hour"]),
        n_before=n_before,
        n_cut=n_cut,
        cut_pct=100.0 * n_cut / n_before,
        cut_weight_pct=100.0 * float(manifest["cut_profile"]["weight_lost_fraction"]),
        rate_pre=r_pre,
        rate_0=r0,
        rate_late=r_late,
        drop=(r_pre - r0) / r_pre,
        chi_drop=1.0 - float(p0.chi_all) / float(pre.chi_all),
        rec=(r_late - r0) / r_pre,
        late_from_h=LATE_HOURS[0],
        late_to_h=LATE_HOURS[1],
    )


def build_control_row(control, cut_hour: float) -> dict:
    """対照 run 1 本を 1 行に。`cut_hour` を切断時刻として損傷 run と同じ式で測る。"""
    m = control.metrics
    r0 = float(_one(m[np.isclose(m.hour, cut_hour)], f"{cut_hour:g} h",
                    control.run_dir).mean_rate_hz)
    r_late = _late_mean(m.hour, m.mean_rate_hz, cut_hour, control.run_dir)
    return dict(
        run=str(control.run_dir),
        seed=int(control.config.simulation.seed),
        backend=control.config.simulation.backend,
        cut_hour=cut_hour,
        rate_0=r0,
        rate_late=r_late,
        rec=(r_late - r0) / r0,
    )


def write_summary(collection, out_dir) -> None:
    """`runs.csv` と `control.csv` を書く。

    種類が `KINDS` に無い損傷 run は表に入れず、その旨を出力する。
    切断時刻が run ごとに違う束は止める (対照を測る時刻が 1 つに決まらない)。

    対照の行の `matches_lesion_baseline` は、対照の r(t0) が同じ (seed, backend) の
    損傷 run の r_pre と一致するか —— 「対照は切断時刻まで損傷 run の親と同じ run」の確認。
    """
    rows = []
    for series in collection.lesion_runs():
        row = build_row(series)
        if row["kind"] is None:
            print(f"  対象外 (種類が決まらない spec): {row['run']}  [{row['spec']}]")
            continue
        rows.append(row)
    if not rows:
        raise ValueError(f"{collection.cond_dir}: 種類の決まる損傷 run がありません")
    cut_hours = {row["cut_hour"] for row in rows}
    if len(cut_hours) != 1:
        raise ValueError(f"切断時刻が run によって違います: {sorted(cut_hours)}")
    cut_hour = cut_hours.pop()

    baseline = {(row["seed"], row["backend"]): row["rate_pre"] for row in rows}
    controls = []
    for control in collection.control_runs():
        row = build_control_row(control, cut_hour)
        ref = baseline.get((row["seed"], row["backend"]))
        row["matches_lesion_baseline"] = None if ref is None else bool(np.isclose(row["rate_0"], ref))
        if row["matches_lesion_baseline"] is False:
            print(f"  注意: 対照の {cut_hour:g} h が損傷 run の baseline と一致しません: {row['run']}")
        controls.append(row)

    out_dir = Path(out_dir)
    write_table(rows, out_dir / RUNS_TABLE_NAME)
    write_table(controls, out_dir / CONTROL_TABLE_NAME)
