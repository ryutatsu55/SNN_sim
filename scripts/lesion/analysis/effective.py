"""実効結合の水準 —— probe ごとの RTE (Watanabe et al. 2026 Fig. 3e)。

「どう測るか」(GTE・z スコア・RTE の式) は `src/utils/analysis/transfer_entropy.py`。
ここが決めるのは**どの窓を基準にし、どの列で残すか**だけ。

**基準は baseline の窓** (切断直前に親 run が記録した窓)。そこで z > 1 だった結合の
集合 C を固定し、各 probe の TE をその窓の TE で割る。`metrics.csv` に入れないのは、
基準の窓が決まらないと値が決まらない (= 全 probe が揃ってから一度に作る) ため ——
`metrics_delta.csv` と同じ扱い。
"""
from __future__ import annotations

from pathlib import Path

from src.utils.analysis.transfer_entropy import (relative_transfer_entropy,
                                                 relevant_connections, transfer_entropy)
from src.utils.runview import MissingData


def _te(window):
    spikes = window.spikes()
    return transfer_entropy(spikes.times, spikes.ids, window.total_neurons,
                            window.record_window_ms)


def rte_rows(series) -> list[dict]:
    """probe ごとの RTE の行。baseline の行は定義上 RTE = 1。

    **TE はフレーム数に依存して偏る**ので、baseline と post の窓幅が違う run では
    比に窓幅の差が混ざる (今の task.yaml はどちらも 10 分)。
    """
    windows = sorted(series.windows, key=lambda window: window.hour)
    baseline = next((window for window in windows if window.is_baseline), None)
    if baseline is None:
        raise MissingData("baseline", "RTE の基準になる baseline の窓がありません")

    te_before = _te(baseline)
    relevant = relevant_connections(te_before)
    rows = []
    for window in windows:
        te = te_before if window is baseline else _te(window)
        result = relative_transfer_entropy(te, te_before, relevant)
        rows.append({
            "probe": int(window.index),
            "phase": window.phase,
            "hours_since_cut": float(window.hour),
            "rte": result.rte,
            "rte_connections": result.num_connections,
            "mean_te_bits": result.mean_te,
        })
    return rows


def write_rte_table(series, out_dir) -> None:
    """`rte_rows()` を `rte.csv` へ書く。"""
    from scripts.lesion.store.formats import RTE_NAME, write_table

    rows = rte_rows(series)
    write_table(rows, Path(out_dir) / RTE_NAME)
    print(f"  RTE: |C| = {rows[0]['rte_connections']} 結合 (baseline で z > 1)")
