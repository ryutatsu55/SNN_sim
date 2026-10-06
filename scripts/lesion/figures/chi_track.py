"""群ごとの同期指標 χ (Golomb) の推移。run 全体で 1 枚。

値は `metrics.csv` の `chi_<群名>` をそのまま読む (再計算しない)。定義は
`src/utils/analysis/synchrony.py` の `golomb_chi`。

**群ごとに底が違う。** 独立に撃っても χ は約 1/√n (n = 群の人数) を取るので、同じ色の
点線でその底を引く。χ が底に近づいたら、その群の同期はほぼ無くなっている。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.utils.analysis.synchrony import CHI_BIN_MS, chi_floor
from src.utils.runview import MissingData
from scripts.lesion.figures import save
from scripts.lesion.figures.style import ALL_GROUP, synchrony_groups

HOUR_COLUMN = "hours_since_cut"
PREFIX = "chi_"

FIGSIZE = (9, 5)
DPI = 200
ALL_COLOR = "black"
GROUP_CMAP = "tab10"
CUT_COLOR = "tab:blue"


def chi_track(series, out_path: Path) -> None:
    """全体と群ごとの χ を、切断からの経過時間に対して描く。

    ベースライン (負の時刻、親 run の窓) の点は post の折れ線と繋がない。
    """
    table = series.metrics().sort_values(HOUR_COLUMN)
    groups = synchrony_groups(series.layout)
    missing = [name for name in groups if f"{PREFIX}{name}" not in table.columns]
    if missing:
        raise MissingData("chi", f"metrics.csv に χ の列がありません: {missing}"
                                 " (この列を持たない古い run です)")
    hours = table[HOUR_COLUMN].to_numpy(dtype=np.float64)
    is_baseline = hours < 0.0
    cmap = plt.get_cmap(GROUP_CMAP)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for index, (name, members) in enumerate(groups.items()):
        values = table[f"{PREFIX}{name}"].to_numpy(dtype=np.float64)
        # 色の 0 番 (tab:blue) は切断線の色なので飛ばす。
        color = ALL_COLOR if name == ALL_GROUP else cmap(index % cmap.N)
        width = 2.0 if name == ALL_GROUP else 1.3
        ax.plot(hours[~is_baseline], values[~is_baseline], marker="o", markersize=3,
                color=color, linewidth=width, label=f"{name} (n={members.size})")
        ax.plot(hours[is_baseline], values[is_baseline], marker="s", markersize=6,
                linestyle="none", color=color)
        ax.axhline(chi_floor(members.size), color=color, linestyle=":", linewidth=0.9)

    ax.axvline(0.0, color=CUT_COLOR, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Time since lesion [h]")
    ax.set_ylabel("χ (Golomb)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Synchrony per group\n"
                 f"(squares = baseline from parent run, dotted = independent-firing floor 1/√n,"
                 f" {CHI_BIN_MS:g} ms bins)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    save(fig, out_path, dpi=DPI)
