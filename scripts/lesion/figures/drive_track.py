"""E / I 駆動の時間発展。run 全体で 1 枚。

値は `metrics.csv` の `drive_*` 列をそのまま読む (再計算しない)。駆動の定義は
`analysis/drive.py`。

左: E / I 由来の駆動を**ベースライン比**で。損傷で入力をどれだけ失い、どれだけ戻したか。
右: E/I 駆動比そのもの。E と I が同じ割合で落ちればここは動かない。
どちらも後ニューロンの極性ごと (E へ / I へ) に線を分ける。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.utils.runview import MissingData
from scripts.lesion.figures import save

HOUR_COLUMN = "hours_since_cut"
TARGETS = (("e", "onto E", "-"), ("i", "onto I", "--"))
SOURCES = (("exc", "from E", "tab:red"), ("inh", "from I", "tab:blue"))
RATIO_COLOR = "tab:purple"

FIGSIZE = (13, 5)
DPI = 200
BASELINE_COLOR = "0.35"
CUT_COLOR = "tab:blue"


def drive_track(series, out_path: Path) -> None:
    """E / I 駆動のベースライン比と E/I 比を、切断をまたいで描く。

    ベースラインは親 run の窓 (負の時刻の 1 点)。比の分母はその点。
    """
    table = series.metrics()
    needed = [f"drive_{source}_onto_{target}_ns" for target, *_ in TARGETS
              for source, *_ in SOURCES]
    missing = [name for name in needed if name not in table.columns]
    if missing:
        raise MissingData("drive", f"metrics.csv に {missing[0]} などの列がありません"
                                   " (駆動を持たない古い run です。replot で作り直せます)")
    table = table.sort_values(HOUR_COLUMN)
    hours = table[HOUR_COLUMN].to_numpy(dtype=np.float64)
    is_baseline = hours < 0.0
    if not is_baseline.any():
        raise MissingData("drive", "ベースラインの行がないので比が取れません")
    base_row = np.flatnonzero(is_baseline)[-1]
    post = ~is_baseline

    fig, (ax_level, ax_ratio) = plt.subplots(1, 2, figsize=FIGSIZE)
    for target, target_label, linestyle in TARGETS:
        for source, source_label, color in SOURCES:
            values = table[f"drive_{source}_onto_{target}_ns"].to_numpy(dtype=np.float64)
            ax_level.plot(hours[post], values[post] / values[base_row], marker="o",
                          markersize=3, color=color, linestyle=linestyle,
                          label=f"{source_label} {target_label}")
        ratio = table[f"drive_ei_ratio_onto_{target}"].to_numpy(dtype=np.float64)
        ax_ratio.plot(hours[post], ratio[post], marker="o", markersize=3,
                      color=RATIO_COLOR, linestyle=linestyle, label=target_label)
        # **親 run から引き継いだ点**なので post の折れ線とは繋がない。
        ax_ratio.plot(hours[base_row], ratio[base_row], marker="s", markersize=7,
                      linestyle="none", color=BASELINE_COLOR)
        ax_ratio.axhline(ratio[base_row], color=BASELINE_COLOR, linestyle=":", linewidth=1.0)

    ax_level.axhline(1.0, color=BASELINE_COLOR, linestyle=":", linewidth=1.0)
    ax_level.set_ylabel("Drive / baseline")
    ax_level.set_title("Synaptic drive relative to baseline")
    ax_ratio.set_ylabel("E / I drive ratio")
    ax_ratio.set_title("E/I drive ratio (square = baseline from parent run)")
    for ax in (ax_level, ax_ratio):
        ax.axvline(0.0, color=CUT_COLOR, linestyle="--", linewidth=1.2)
        ax.set_xlabel("Time since lesion [h]")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Time-averaged synaptic conductance received per neuron (dashed line = lesion)")
    save(fig, out_path, dpi=DPI)
