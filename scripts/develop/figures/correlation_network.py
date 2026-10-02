"""空間に置いたニューロン間相関のマップ (Watanabe et al. 2026 Fig. 3c 上段)。

`network_sample` と同じ空間配置 (体細胞の位置・エリアの境界線) の上に、相関の強い
ニューロン対を**無向の線**で重ね、相関係数で色を付ける。相関の定義は
`src/utils/analysis/correlation.py`。

## 描くペアの選び方

**全ペアのうち相関が `MIN_CORRELATION` 以上のものだけ。** 結合の有無も距離も見ない ——
機能的結合の図なので、線が引かれていること自体が「強く同期している」の意味になる。

閾値は絶対値で固定する。窓ごとに分位点で決めると、系全体の相関が変わっても同じ本数が
残ってしまい、**同期が育っていく過程が図に出なくなる**。

## `weight_network` との読み分け

同じ空間配置に別のものを描いた 2 枚。`weight_network` は**シナプスがあるか** (構造)、
こちらは**一緒に発火しているか** (機能)。線が重ならないのは異常ではない ——
直接繋がっていなくても共通入力で同期するし、繋がっていても効いていない結合はある。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from src.utils.analysis.correlation import RATE_BIN_MS, neuronal_correlation
from src.utils.runview import optional
from scripts.develop.figures import save
from scripts.develop.figures.area import draw_area
# 空間配置は `network_sample` と同じもの。実装を 2 つ持たないよう借りる。
from scripts.develop.figures.network import (NODE_SIZE, apply_axis_limits, draw_nodes,
                                            excitatory_mask)

# **これ以上の相関のペアだけを線にする。** 閾値を変えると図の密度が大きく変わるので、
# 窓をまたいで比べるあいだは動かさないこと。
MIN_CORRELATION = 0.4

# 見た目。**色は論文 Fig. 3c と同じ viridis**。範囲は閾値から 1 まで —— [0, 1] に取ると
# 描いた線が全部カラーバーの上半分に固まり、強弱が読めなくなる。
FIGSIZE = (12, 10)
VMAX = 1.0
CMAP = "viridis"
DPI = 300
EDGE_WIDTH = 0.5


def correlation_network(window, out_path: Path) -> None:
    """記録窓 1 つぶんの相関を、体細胞の位置の上に無向グラフとして描く。

    **描くのは相関が `MIN_CORRELATION` 以上のペアだけ。** 線の本数そのものが
    「強く同期しているペアがどれだけあるか」を表すので、2 枚を比べるときは色だけで
    なく密度も見ること。
    """
    config, layout = window.config, window.layout
    coords = np.asarray(window.coords())
    area = optional(window.area)
    x, y = coords[:, 0], coords[:, 1]
    total = coords.shape[0]

    spikes = window.spikes()
    matrix = neuronal_correlation(spikes.times, spikes.ids, total,
                                  window.record_window_ms)
    # 無向なので上三角だけ。`nan` (窓の中で 1 発も撃たないニューロン) は比較で落ちる。
    rows, cols = np.nonzero(np.triu(matrix >= MIN_CORRELATION, 1))
    values = matrix[rows, cols]
    # 相関の低い線から描く。重なったとき強いペアが上に来る。
    order = np.argsort(values)
    rows, cols, values = rows[order], cols[order], values[order]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # 塗りは入れない (`network` と同じ理由 —— part ごとの塗り分けはエッジと色が競合する)。
    area_drawn = draw_area(ax, area, fill=False, boundary=True, zorder=0)

    segments = np.stack([np.column_stack([x[rows], y[rows]]),
                         np.column_stack([x[cols], y[cols]])], axis=1)
    lines = LineCollection(segments, array=values, cmap=CMAP,
                           norm=plt.Normalize(MIN_CORRELATION, VMAX),
                           linewidths=EDGE_WIDTH, zorder=1)
    ax.add_collection(lines)
    # ニューロンは**サンプリングしない**。閾値で絞るのは線であって、位置は全部見せる。
    draw_nodes(ax, x, y, np.arange(total), excitatory_mask(layout, total), NODE_SIZE)

    ax.set_aspect("equal")
    ax.set_title(f"correlation_network {window.hour:g} h\n"
                 f"{total} neurons, {values.size} pairs with"
                 f" correlation >= {MIN_CORRELATION:g} ({RATE_BIN_MS:g} ms bins)")
    ax.set_xlabel("X Coordinate [um]")
    ax.set_ylabel("Y Coordinate [um]")
    apply_axis_limits(ax, config, area_drawn)
    fig.colorbar(lines, ax=ax, fraction=0.046, pad=0.04, label="correlation")

    save(fig, out_path, dpi=DPI, bbox_inches="tight")
