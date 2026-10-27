"""結合 1 本 = 1 点の散布図を、pre / post の極性 (E→E / E→I / I→E / I→I) で色分けして描く。

`draw_block_scatter()` が見た目 (色・平均線・凡例) を 1 か所で持つ。run 全体の散布図
(`change.py`) と probe ごとの散布図 (`correlation_weight_scatter`) が同じ描き方になるように。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.utils.analysis.correlation import RATE_BIN_MS, neuronal_correlation
from src.utils.analysis.weights import (BLOCK_ORDER, align_subset_to_coo, block_masks,
                                        excitatory_flags)
from scripts.lesion.figures import save, style

FIGSIZE = (10, 6)
DPI = 200
POINT_SIZE = 10
POINT_ALPHA = 0.8
# 変化量の軸に引く 0 の線。
ZERO_LINE_COLOR = "0.5"
# 平均の破線。群ごとは群の色、全体はこの色。点より下に薄く引く。
# グリッド (`GRID_STYLE`) と見分けられるよう、こちらは太く間隔の長い破線にする。
MEAN_LINE_STYLE = dict(linestyle=(0, (8, 4)), linewidth=1.6, alpha=0.45)
# グリッドは細い実線。平均の破線と同じ見た目にしない。
GRID_STYLE = dict(linestyle="-", linewidth=0.4, color="0.85")
MEAN_LINE_ZORDER = 0.5
ALL_MEAN_COLOR = "black"


def block_label(name: str) -> str:
    """`EE` → `E→E`。"""
    return f"{name[0]}→{name[1]}"


def draw_block_scatter(layout, out_path: Path, row, col, x, y, *, xlabel: str, ylabel: str,
                       title: str, zero_x: bool = False, zero_y: bool = False,
                       mean_x: str | None = None, mean_y: str | None = None,
                       legend_means: dict | None = None) -> None:
    """結合 1 本 = 1 点の散布図を、pre / post の極性 (`BLOCK_ORDER`) で色分けして描く。

    x / y のどちらかが nan の点は描かない (本数にも平均にも数えない)。`mean_x` / `mean_y` に
    凡例用の短い名前を渡した軸だけ、群ごと (群の色) と全体 (黒) の平均を破線で引き、
    値を凡例に書く。`legend_means` (名前 → 結合ごとの値) は線を引かずに平均を凡例にだけ書く。
    凡例は軸の右外に置く。
    """
    row, col = np.asarray(row, dtype=np.int64), np.asarray(col, dtype=np.int64)
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    masks = block_masks(row, col, excitatory_flags(layout, int(layout.total_neurons)))
    groups = {"all": finite, **{name: masks[name] & finite for name in BLOCK_ORDER}}
    colors = {"all": ALL_MEAN_COLOR, **style.BLOCK_COLORS}
    means = [(name, values) for name, values in ((mean_x, x), (mean_y, y)) if name is not None]
    means += [(name, np.asarray(values, dtype=np.float64))
              for name, values in (legend_means or {}).items()]

    def label(name, keep) -> str:
        head = "all" if name == "all" else block_label(name)
        parts = [f"{key}={np.nanmean(values[keep]):.3g}" for key, values in means]
        tail = f", mean {', '.join(parts)}" if parts and keep.any() else ""
        return f"{head} (n={int(keep.sum())}{tail})"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if zero_x:
        ax.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=0.8, zorder=0)
    if zero_y:
        ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=0.8, zorder=0)
    handles = []
    for name, keep in groups.items():
        if name == "all":
            if mean_x is not None or mean_y is not None:
                handles.append(Line2D([], [], color=ALL_MEAN_COLOR, label=label(name, keep),
                                      **MEAN_LINE_STYLE))
            elif means:
                handles.append(Line2D([], [], linestyle="none", label=label(name, keep)))
        else:
            handles.append(ax.scatter(x[keep], y[keep], s=POINT_SIZE, color=colors[name],
                                      alpha=POINT_ALPHA, edgecolors="none", zorder=1,
                                      label=label(name, keep)))
        if keep.any():
            if mean_x is not None:
                ax.axvline(x[keep].mean(), color=colors[name], zorder=MEAN_LINE_ZORDER,
                           **MEAN_LINE_STYLE)
            if mean_y is not None:
                ax.axhline(y[keep].mean(), color=colors[name], zorder=MEAN_LINE_ZORDER,
                           **MEAN_LINE_STYLE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{int(finite.sum())} of {row.size} synapses", fontsize=10)
    ax.grid(True, zorder=0, **GRID_STYLE)
    ax.set_axisbelow(True)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
              markerscale=3.0, fontsize=8)
    save(fig, out_path, tight_layout=False, dpi=DPI, bbox_inches="tight")


def correlation_weight_scatter(window, out_path: Path) -> None:
    """切断後の結合 1 本 = 1 点で、横 = その窓の相関、縦 = その窓の重みを描く。

    baseline の窓でも点は切断後の結合だけ (切断前の重みを (pre, post) で引き当てる)。
    重みは窓の先頭の値。相関が未定義のペアは描かない。
    """
    total = int(window.layout.total_neurons)
    kept = window.wiring_after_cut()
    row = np.asarray(kept.row, dtype=np.int64)
    col = np.asarray(kept.col, dtype=np.int64)
    weights = np.asarray(window.weights(), dtype=np.float64)
    if window.is_baseline:
        pre = window.wiring()
        weights = align_subset_to_coo(pre.row, pre.col, weights, row, col, total)
    spikes = window.spikes()
    correlation = neuronal_correlation(spikes.times, spikes.ids, total,
                                       window.record_window_ms)[row, col]
    draw_block_scatter(
        window.layout, out_path, row, col, correlation, weights,
        xlabel=f"correlation ({RATE_BIN_MS:g} ms bins)", ylabel="weight",
        title=f"Correlation vs weight (synapses kept after the lesion)\n{window.label}",
        mean_x="corr", mean_y="w")
