"""損傷実験で probe 1 つぶんの指標をまとめる。

**本番の run も `replot.py` の再解析も、必ず `build_row()` を通る。** 列が 1 本の関数で
決まるので、本番と再解析で食い違いようがない (develop の `analysis/metrics.py` と同じ形)。

「何を測るか」は実験ごとに違うのでここ (実験のディレクトリ) が持ち、「どう測るか」は
`src/utils/analysis/` の既存関数に任せる。この分担のおかげで、ここにあるのは呼び出しと
列名の組み立てだけになる。**数式は分岐したらバグ**なので共有層のまま。

`metrics.csv` に入るのは**絶対値だけ**。ベースラインとの差 (`delta_*` / `ratio_*`) は
そこから導ける派生値なので、run 終了後に `metrics_delta.csv` として別に出す
(`write_delta_table`)。1 行ずつ追記する記録に、全行が揃わないと決まらない列は混ぜない。
"""
from __future__ import annotations

import numpy as np

from src.utils.analysis.avalanche import split_avalanches
from src.utils.analysis.criticality import (bimodality_d, burstiness_index,
                                            criticality_index_delta_cr)
from src.utils.analysis.connectivity import group_connection_probability
from src.utils.analysis.graph import (betweenness_centrality, classify_roles,
                                      degree_table, graph_metrics, module_participation)
from src.utils.analysis.isi import isi_metrics
from src.utils.analysis.powerlaw import log_likelihood_ratio_power_vs_exponential
from src.utils.analysis.spikes import diagnose_activity, firing_rates, spike_group_metrics
from src.utils.analysis.weights import (block_values, excitatory_flags,
                                        synapse_distances, weight_block_metrics)
from src.utils.analysis.avalanche import analyze_avalanches


def resolve_avalanche_smax(config) -> int:
    """べき乗フィット / ΔCr の上限サイズ。**システムサイズ N。**

    論文 (Ikeda-Akita-Takahashi 2023) の [1, 100] は N=100 のネットワークの話で、100 は
    定数ではなく系のサイズそのもの。smax は「データの切り取り」ではなく**モデルの
    正規化台** (p(s) = s^-α / Σ_{k=1}^{smax} k^-α) なので、観測サイズが smax を
    超えなくても smax を変えれば α は動く。

    **親 run と同じ値でなければ比較できない。** 親も N から決めているので、同じ
    ネットワークを引き継ぐ限り自動的に一致する。
    """
    return int(config.simulation.N)


def build_row(window) -> dict:
    """`metrics.csv` の 1 行を作る。**本番の run も再解析もこの 1 つを通る。**

    必要な値は全部 `window` から取る (smax も解析オプションも config から導けるので
    引数にしない)。`spikes().times` は**窓の先頭を 0 とするローカル時刻**で、絶対時刻を
    使うとアバランチ分割は同じでも burstiness のビン割りがずれる (契約でローカルに固定済み)。

    重い指標 (媒介中心性・クラスタ係数) の on/off は `task.*` にある。**結果に効くので
    引数ではなく config 側**にあり、run ディレクトリの `config.yaml` を見れば
    「その run が何を測ったか」が分かる。
    """
    protocol = window.config.task
    spikes = window.spikes()
    coo = window.coo()
    out: dict = {
        "probe": int(window.index),
        "phase": window.phase,
        "hours_since_cut": float(window.hour),
        "window_ms": float(window.record_window_ms),
    }
    out.update(spike_metrics(
        spikes.times, spikes.ids, window.layout, window.total_neurons,
        window.record_window_ms,
        smax=resolve_avalanche_smax(window.config),
        dt_ms=float(window.config.simulation.dt),
    ))
    out.update(structure_metrics(
        coo.row, coo.col, coo.weights, window.layout, window.total_neurons,
        include_betweenness=bool(getattr(protocol, "include_betweenness", True)),
        include_clustering=bool(getattr(protocol, "include_clustering", True)),
        hub_z=float(getattr(protocol, "hub_z", 2.5)),
    ))
    return add_diagnosis(out)


def spike_metrics(times, ids, layout, num_neurons: int, window_ms: float,
                  smax: int, dt_ms: float, *, include_avalanche_suite: bool = True) -> dict:
    """1 記録窓ぶんの神経科学指標。

    `analyze_avalanches` (ビン分割) と `split_avalanches` (ギャップ分割) の**両方**を
    走らせる。前者は分岐パラメータ σ_bp まで出る一方でビン連結方式、後者は親 run の
    `develop.py` と同じギャップ分割方式で、ΔCr / LLR / D はこちらで測らないと
    **損傷前の親 run の値と比較できない**。
    """
    times = np.asarray(times, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)

    rates = firing_rates(ids, num_neurons, window_ms)
    avalanche = split_avalanches(times)
    row = {
        "num_spikes": int(times.size),
        "mean_rate_hz": float(np.mean(rates)),
        "median_rate_hz": float(np.median(rates)),
        "max_rate_hz": float(np.max(rates)) if rates.size else float("nan"),
        # --- Akita 系 (親 run と同じ切り出し方。ここが比較の軸) ---
        "avalanche_threshold_ms": avalanche.threshold_ms,
        "num_avalanches": int(avalanche.sizes.size),
        "avalanche_smax": int(smax),
        "llr": log_likelihood_ratio_power_vs_exponential(avalanche.sizes),
        "delta_cr": criticality_index_delta_cr(avalanche.sizes, smax=smax),
        "burstiness_index": burstiness_index(times, window_ms),
        "bimodality_d": bimodality_d(avalanche.sizes),
    }
    polarity = layout.ids_by("polarity")
    row.update(spike_group_metrics(ids, polarity.get("excitatory"), polarity.get("inhibitory"),
                                   window_ms))
    row.update(isi_metrics(times, ids, num_neurons, window_ms))

    if include_avalanche_suite:
        # ビン連結方式。σ_bp (分岐パラメータ) は臨界性の回復を最も直接に示す量。
        _av, _lags, _corr, suite = analyze_avalanches(
            times, ids, window_ms, min_bin_ms=dt_ms, size_fit_max=smax,
        )
        row.update({f"bp_{key}": value for key, value in suite.items()})
    return row


def structure_metrics(row, col, weights, layout, num_neurons: int, *,
                      module_axis: str = "module",
                      include_betweenness: bool = True,
                      include_clustering: bool = True,
                      hub_z: float = 2.5) -> dict:
    """1 時点ぶんの構造指標 (トポロジー + 重み)。"""
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)

    labels = layout.labels(module_axis) if layout.has_axis(module_axis) else None
    out = graph_metrics(row, col, num_neurons, weights=weights, labels=labels,
                        include_betweenness=include_betweenness,
                        include_clustering=include_clustering, hub_z=hub_z)

    if labels is not None:
        group = group_connection_probability(row, col, layout, module_axis)
        out["within_module_prob"] = float(group.within_probability)
        out["between_module_prob"] = float(group.between_probability)
        out["segregation"] = float(group.segregation)

    blocks = block_values(weights, row, col, layout)
    out.update(weight_block_metrics(blocks, wmax=1.0))
    return out


def add_diagnosis(row: dict) -> dict:
    """`develop.py` と同じ活動診断列を足す。"""
    row.update(diagnose_activity(
        mean_rate_hz=row.get("mean_rate_hz", float("nan")),
        weight_at_max_fraction=row.get("weight_at_max_fraction", float("nan")),
    ))
    return row


def delta_from_baseline(rows: list[dict], baseline: dict) -> list[dict]:
    """各行に `delta_{col}` / `ratio_{col}` を足す。

    ベースラインは**切断直前の 1 点のみ** (sham を作らない方針)。よって
    「回復」と「損傷が無くても進んだ発達の続き」は原理的に分離できない。
    親 run の終盤のドリフト幅 (`lesion.json` の `parent_drift`) と比べて、
    同オーダーなら結論を出さないこと。
    """
    out = []
    for row in rows:
        merged = dict(row)
        for key, value in row.items():
            base = baseline.get(key)
            if not isinstance(value, (int, float)) or not isinstance(base, (int, float)):
                continue
            if isinstance(value, bool) or isinstance(base, bool):
                continue
            merged[f"delta_{key}"] = float(value) - float(base)
            merged[f"ratio_{key}"] = (float(value) / float(base)) if base else float("nan")
        out.append(merged)
    return out


DELTA_NAME = "metrics_delta.csv"


def write_delta_table(series, out_dir) -> None:
    """`metrics.csv` からベースラインとの差の表を作り、`metrics_delta.csv` へ書く。

    **run 全体が揃って初めて決まる**ので、1 行ずつ追記する `metrics.csv` には混ぜない。
    基準にするのは phase が baseline の行 (無ければ最初の行)。
    """
    from pathlib import Path

    from scripts.lesion.store.records import PHASE_BASELINE, write_table

    rows = series.metrics().to_dict("records")
    if not rows:
        return
    baseline = next((row for row in rows if row.get("phase") == PHASE_BASELINE), rows[0])
    write_table(delta_from_baseline(rows, baseline), Path(out_dir) / DELTA_NAME)


# ======================================================================================
# 切断したものの素性 — 「どんな特徴の結合を、どれくらい切ったか」
# ======================================================================================
#
# 切断本数だけでは実験の記述にならない。同じ 115 本でも、ハブの出力を集中的に落としたのか、
# 弱い結合を薄く広く落としたのかで意味が違う。ここでは **切断群と残存群を同じ土俵で並べる**
# ことに徹する —— 「切ったものの平均重みが 0.62」より「切ったものは 0.62、残ったものは 0.38」
# のほうが、何を失ったかを一意に決める。

_ROLE_ORDER = ("provincial_hub", "connector_hub", "kinless_hub",
               "non_hub_connector", "peripheral")


def _stats(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "sd": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"n": int(finite.size), "mean": float(np.mean(finite)),
            "median": float(np.median(finite)), "sd": float(np.std(finite)),
            "min": float(np.min(finite)), "max": float(np.max(finite))}


def cut_profile(coo, cut_mask, layout, num_neurons: int, *, module_axis: str = "module",
                coords=None, crossed=None, crossed_part_names=None,
                include_betweenness: bool = True,
                hub_z: float = 2.5) -> tuple[dict, list[dict], dict]:
    """切断されたシナプスの素性を、残存シナプスと対比して記述する。

    ニューロン単位の指標 (次数・participation・ハブ役割) は**切断前のネットワーク**で
    測る = 「切った時点でそれがどういう位置にいたか」。

    Returns:
        (per_synapse, comparison, summary)
        per_synapse: 切断されたシナプスごとの属性 (npz に落とす形の配列辞書)
        comparison : 属性ごとの「切断群 vs 残存群」の統計 (CSV の行)
        summary    : lesion.json に入れる要約
    """
    cut = np.asarray(cut_mask, dtype=bool)
    kept = ~cut
    row = np.asarray(coo.row, dtype=np.int64)
    col = np.asarray(coo.col, dtype=np.int64)
    weights = np.asarray(coo.weights, dtype=np.float64)
    delays = np.asarray(coo.delays, dtype=np.float64)

    # --- ニューロン単位の指標 (切断前) -------------------------------------------
    degrees = degree_table(row, col, num_neurons, weights=weights)
    labels = (np.asarray(layout.labels(module_axis))
              if layout.has_axis(module_axis) else None)
    if labels is not None:
        part = module_participation(row, col, num_neurons, labels)
        roles = classify_roles(part["participation"], part["within_module_z"], hub_z=hub_z)
    else:
        part = {"participation": np.zeros(num_neurons), "within_module_z": np.zeros(num_neurons)}
        roles = np.full(num_neurons, "unknown", dtype=object).astype(str)
    betweenness, betweenness_exact = (
        betweenness_centrality(row, col, num_neurons) if include_betweenness else (None, False))

    is_exc = excitatory_flags(layout, num_neurons)
    block_of = np.where(is_exc[row], np.where(is_exc[col], "EE", "EI"),
                        np.where(is_exc[col], "IE", "II"))

    # --- シナプス単位の属性 ------------------------------------------------------
    attributes = {
        "weight": weights,
        "delay_ms": delays,
        "pre_out_degree": degrees["out_degree"][row],
        "pre_in_degree": degrees["in_degree"][row],
        "post_out_degree": degrees["out_degree"][col],
        "post_in_degree": degrees["in_degree"][col],
        "pre_out_strength": degrees["out_strength"][row],
        "post_in_strength": degrees["in_strength"][col],
        "pre_participation": part["participation"][row],
        "post_participation": part["participation"][col],
        "pre_within_module_z": part["within_module_z"][row],
        "post_within_module_z": part["within_module_z"][col],
    }
    if betweenness is not None:
        attributes["pre_betweenness"] = betweenness[row]
        attributes["post_betweenness"] = betweenness[col]
    if coords is not None:
        attributes["distance_um"] = synapse_distances(coords, row, col)

    # --- 切断群 vs 残存群 --------------------------------------------------------
    comparison = []
    for name, values in attributes.items():
        cut_stats, kept_stats = _stats(values[cut]), _stats(values[kept])
        ratio = (cut_stats["mean"] / kept_stats["mean"]
                 if kept_stats["mean"] not in (0.0,) and np.isfinite(kept_stats["mean"])
                 else float("nan"))
        comparison.append({
            "attribute": name,
            "cut_n": cut_stats["n"], "kept_n": kept_stats["n"],
            "cut_mean": cut_stats["mean"], "kept_mean": kept_stats["mean"],
            "ratio_cut_over_kept": ratio,
            "cut_median": cut_stats["median"], "kept_median": kept_stats["median"],
            "cut_sd": cut_stats["sd"], "kept_sd": kept_stats["sd"],
            "cut_min": cut_stats["min"], "cut_max": cut_stats["max"],
        })

    # --- 分類ごとの内訳 ----------------------------------------------------------
    # 「その群の結合を何割切ったか」まで出す。本数だけだと、元から多い群が上位に来る。
    def _breakdown(values_per_synapse, keys):
        out = {}
        for key in keys:
            member = values_per_synapse == key
            total = int(member.sum())
            if total == 0:
                continue
            n_cut = int((member & cut).sum())
            out[str(key)] = {"total": total, "cut": n_cut,
                             "fraction_of_group_cut": n_cut / total,
                             "share_of_cut": n_cut / max(int(cut.sum()), 1)}
        return out

    summary = {
        "num_cut": int(cut.sum()),
        "num_kept": int(kept.sum()),
        "fraction_cut": float(cut.mean()),
        "weight_lost_fraction": (float(weights[cut].sum() / weights.sum())
                                 if weights.sum() > 0 else float("nan")),
        "betweenness_exact": bool(betweenness_exact),
        "by_ei_block": _breakdown(block_of, ("EE", "EI", "IE", "II")),
        "by_pre_role": _breakdown(roles[row], _ROLE_ORDER),
        "by_post_role": _breakdown(roles[col], _ROLE_ORDER),
    }
    if labels is not None:
        same_module = labels[row] == labels[col]
        summary["by_module_relation"] = _breakdown(
            np.where(same_module, "within_module", "between_module"),
            ("within_module", "between_module"))
        summary["by_pre_module"] = _breakdown(labels[row], np.unique(labels))
    if crossed is not None and crossed_part_names is not None:
        crossed = np.asarray(crossed, dtype=bool)
        summary["by_crossed_bridge"] = {
            str(name): {"total": int(crossed[:, j].sum()),
                        "cut": int((crossed[:, j] & cut).sum()),
                        "fraction_of_group_cut": (
                            float((crossed[:, j] & cut).sum() / crossed[:, j].sum())
                            if crossed[:, j].sum() else float("nan"))}
            for j, name in enumerate(crossed_part_names)
        }

    # --- 切断シナプスごとの記録 (npz) --------------------------------------------
    per_synapse = {"row": row[cut], "col": col[cut],
                   "ei_block": block_of[cut].astype(str),
                   "pre_role": np.asarray(roles)[row][cut].astype(str),
                   "post_role": np.asarray(roles)[col][cut].astype(str)}
    per_synapse.update({name: values[cut] for name, values in attributes.items()})
    if labels is not None:
        per_synapse["pre_module"] = labels[row][cut].astype(str)
        per_synapse["post_module"] = labels[col][cut].astype(str)
    if crossed is not None:
        per_synapse["crossed_parts"] = np.asarray(crossed, dtype=bool)[cut]
    return per_synapse, comparison, summary


def format_cut_profile(comparison: list[dict], summary: dict, *, precision: int = 4) -> str:
    """端末に出す 1 枚の表。実行ログを見れば何を切ったかが分かるようにする。"""
    lines = [
        f"切断 {summary['num_cut']} / {summary['num_cut'] + summary['num_kept']} 本 "
        f"({summary['fraction_cut'] * 100:.1f}%)、"
        f"総重みの {summary['weight_lost_fraction'] * 100:.1f}% を喪失",
        "",
        f"  {'属性':<22}{'切断群':>10}{'残存群':>10}{'比':>8}",
        "  " + "-" * 50,
    ]
    for row in comparison:
        lines.append(f"  {row['attribute']:<22}{row['cut_mean']:>10.{precision}g}"
                     f"{row['kept_mean']:>10.{precision}g}"
                     f"{row['ratio_cut_over_kept']:>8.2f}")
    for title, key in (("E/I ブロック別", "by_ei_block"),
                       ("pre のハブ役割別", "by_pre_role"),
                       ("モジュール関係別", "by_module_relation")):
        section = summary.get(key)
        if not section:
            continue
        lines += ["", f"  {title} (その群の結合のうち何本を切ったか)"]
        for name, stats in section.items():
            lines.append(f"    {name:<20}{stats['cut']:>6} / {stats['total']:<6}"
                         f" = {stats['fraction_of_group_cut'] * 100:5.1f}%"
                         f"   (切断全体の {stats['share_of_cut'] * 100:5.1f}%)")
    return "\n".join(lines)
