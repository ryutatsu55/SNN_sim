"""実効結合の水準 (RTE) の推移 (Watanabe et al. 2026 Fig. 3e)。run 全体で 1 枚。

値は `rte.csv` をそのまま読む (`analysis/effective.py` が書く)。定義は
`src/utils/analysis/transfer_entropy.py`。

上段が RTE (損傷前に有意だった結合の集合 C の上で、TE を損傷前の値で割った比の平均)、
下段が全ペアの平均 TE。**RTE は C に入っていない結合を見ない**ので、損傷後に生まれた
経路は上段には出ない。下段が上段より戻っていれば、元の経路以外で情報が流れている。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.utils.analysis.transfer_entropy import RELEVANT_Z, TE_BIN_MS
from scripts.lesion.figures import save

HOUR_COLUMN = "hours_since_cut"

FIGSIZE = (9, 6)
DPI = 200
POST_COLOR = "tab:green"
BASELINE_COLOR = "0.35"
CUT_COLOR = "tab:blue"


def rte_track(series, out_path: Path) -> None:
    """RTE と全ペアの平均 TE を、切断からの経過時間に対して 2 段で描く。"""
    table = series.rte().sort_values(HOUR_COLUMN)
    hours = table[HOUR_COLUMN].to_numpy(dtype=np.float64)
    is_baseline = hours < 0.0
    connections = int(table["rte_connections"].iloc[0])

    fig, (ax_rte, ax_te) = plt.subplots(2, 1, sharex=True, figsize=FIGSIZE)
    for ax, column, ylabel in ((ax_rte, "rte", "RTE"),
                               (ax_te, "mean_te_bits", "mean TE, all pairs [bit]")):
        values = table[column].to_numpy(dtype=np.float64)
        ax.plot(hours[~is_baseline], values[~is_baseline], marker="o", markersize=4,
                color=POST_COLOR, linewidth=1.6, label="post-lesion")
        ax.plot(hours[is_baseline], values[is_baseline], marker="s", markersize=7,
                linestyle="none", color=BASELINE_COLOR, label="baseline (from parent run)")
        if is_baseline.any():
            ax.axhline(float(values[is_baseline][-1]), color=BASELINE_COLOR,
                       linestyle=":", linewidth=1.0)
        ax.axvline(0.0, color=CUT_COLOR, linestyle="--", linewidth=1.2)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle="--", alpha=0.4)
    ax_rte.legend(loc="best", fontsize=8)
    ax_rte.set_title(f"Relative transfer entropy\n|C| = {connections} connections"
                     f" with z > {RELEVANT_Z:g} at baseline, {TE_BIN_MS:g} ms frames",
                     fontsize=10)
    ax_te.set_xlabel("Time since lesion [h]")
    save(fig, out_path, dpi=DPI)
