"""**run 終了後**に出すもの。全記録窓をまとめて見る図。

本番の run の末尾も、再解析もここを通る。以前は `run_one` が 3 枚、`replot` が 2 枚を
それぞれ別に呼んでいたので、再解析すると `metrics.csv` は新しいのに
`weight_matrix_*.png` だけ古い、という状態が同じ `data/` の中に同居していた。

1 枚描けなくても止めない。ここへ来る時点で npz も `metrics.csv` も確定しているので、
**数時間回した結果を描画の都合で失わない** (`panels.py` と同じ方針)。
"""
from __future__ import annotations

from scripts.develop.report import guard
from scripts.develop.store import paths
from scripts.develop.figures.fig2c import fig2c
from scripts.develop.figures.fig2d import fig2d
from scripts.develop.figures.weight_matrix import (weight_delta_panel, weight_matrix,
                                                   weight_panel)

FIG2C_NAME = "figure2c_reproduction.png"
FIG2D_NAME = "figure2d_firing_rate_scatter.png"
PANEL_NAME = "weight_matrix_panel.png"
DELTA_PANEL_NAME = "weight_delta_panel.png"


def emit(series) -> None:
    """run 全体の図を `<run>/figures/overview/` へ。"""
    out = paths.fig_dir(series.run_dir, paths.OVERVIEW)
    out.mkdir(parents=True, exist_ok=True)

    guard("Figure 2c", fig2c, series, out / FIG2C_NAME)
    guard("Figure 2d", fig2d, series, out / FIG2D_NAME)

    # 重み行列は「1 時刻 1 枚」と「時系列を並べたパネル」の両方を出す。前者は 1 枚を
    # 拡大して読むため、後者は変化の向きを一目で見るため。
    for window in series.windows:
        guard(f"weight matrix {window.hour:g}h", weight_matrix, window,
              out / f"weight_matrix_{window.hour:g}h.png")
    guard("weight matrix panel", weight_panel, series, out / PANEL_NAME)
    guard("weight delta panel", weight_delta_panel, series, out / DELTA_PANEL_NAME)
