"""**run 終了後**に出すもの。全記録窓をまとめて見る図。

本番の run の末尾も再解析もここを通るので、どちらから作っても同じ図が揃う。

1 枚描けなくても止めない。ここへ来る時点で npz も `metrics.csv` も確定しているので、
**回した結果を描画の都合で失わない。**
"""
from __future__ import annotations

from scripts.akita_soc.report import guard
from scripts.akita_soc.store import paths
from scripts.akita_soc.figures.fig2c import fig2c
from scripts.akita_soc.figures.fig2d import fig2d


# **この段階で出るものの一覧。** 足すならここへ 1 行足す。
#
# `(ログ上の名前, 描く関数, ファイル名)`。置き場所は全部 `report/overview/` なので
# 表には持たせない。渡す view は run 全体の `Series`。
FIGURES = (
    ("Figure 2c", fig2c, "figure2c_reproduction.png"),
    ("Figure 2d", fig2d, "figure2d_firing_rate_scatter.png"),
)


# 同じ段階で出す**報告の表**。形と回し方は `FIGURES` と同じ。今は無い。
TABLES = ()


def emit(series) -> None:
    """run 全体の図と表を `<run>/report/overview/` へ。"""
    out = paths.report_dir(series.run_dir, paths.OVERVIEW)
    out.mkdir(parents=True, exist_ok=True)

    for label, make, name in FIGURES + TABLES:
        guard(label, make, series, out / name)
