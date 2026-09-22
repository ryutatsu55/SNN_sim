"""**run 終了後**に出すもの。全記録窓をまとめて見る図。

本番の run の末尾も再解析もここを通るので、どちらから作っても同じ図が揃う。

1 枚描けなくても止めない。ここへ来る時点で npz も `metrics.csv` も確定しているので、
**回した結果を描画の都合で失わない。**
"""
from __future__ import annotations

from scripts.develop.report import guard
from scripts.develop.store import paths
from scripts.develop.figures.fig2c import fig2c
from scripts.develop.figures.fig2d import fig2d
from scripts.develop.figures.weight_matrix import (weight_delta_panel, weight_matrix,
                                                   weight_panel)


# **この段階で出るものの一覧。** 足すならここへ 1 行足す。
#
# `(ログ上の名前, 描く関数, ファイル名)`。置き場所は全部 `figures/overview/` なので
# 表には持たせない。渡す view は run 全体の `Series`。
FIGURES = (
    ("Figure 2c", fig2c, "figure2c_reproduction.png"),
    ("Figure 2d", fig2d, "figure2d_firing_rate_scatter.png"),
    ("weight matrix panel", weight_panel, "weight_matrix_panel.png"),
    ("weight delta panel", weight_delta_panel, "weight_delta_panel.png"),
)

# **記録時刻ごとに 1 枚ずつ**出るもの。渡す view が `Series` ではなく `Window` なので、
# 上の表とは別に持つ。
#
# 重み行列を「1 時刻 1 枚」と「時系列を並べたパネル」の両方で出しているのは、前者が
# 1 枚を拡大して読むため、後者が変化の向きを一目で見るため。
PER_WINDOW_FIGURES = (
    ("weight matrix", weight_matrix, "weight_matrix_{tag}.png"),
)


def emit(series) -> None:
    """run 全体の図を `<run>/figures/overview/` へ。"""
    out = paths.fig_dir(series.run_dir, paths.OVERVIEW)
    out.mkdir(parents=True, exist_ok=True)

    for label, draw, name in FIGURES:
        guard(label, draw, series, out / name)

    for window in series.windows:
        tag = f"{window.hour:g}h"
        for label, draw, template in PER_WINDOW_FIGURES:
            guard(f"{label} {tag}", draw, window, out / template.format(tag=tag))
