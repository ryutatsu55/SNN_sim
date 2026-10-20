"""**run 終了後**に出すもの。全 probe をまとめて見る図。

本番の run の末尾も、再解析もここを通る。1 枚描けなくても止めない —— ここへ来る時点で
npz も `metrics.csv` も確定しているので、**回復過程を丸ごと回した結果を描画の都合で
失わない**。
"""
from __future__ import annotations

from scripts.lesion.report import guard
from scripts.lesion.store import paths
from scripts.lesion.analysis.effective import write_rte_table
from scripts.lesion.figures.chi_track import chi_track
from scripts.lesion.figures.correlation_track import mean_correlation_track
from scripts.lesion.figures.drive_track import drive_track
from scripts.lesion.figures.firing_rate import firing_rate_scatter
from scripts.lesion.figures.participation_heatmap import participation_heatmap
from scripts.lesion.figures.rte_track import rte_track
from scripts.lesion.figures.weight_track import (weight_distribution_shift,
                                                 weight_trajectories)


# **この段階で出るものの一覧。** 足すならここへ 1 行足す。
#
# `(ログ上の名前, 描く関数, ファイル名)`。置き場所は全部 `report/overview/` なので
# 表には持たせない。渡す view は run 全体の `Series`。
#
# **記録時刻ごとに出る図はこの段階には無い。** probe ごとの図は `panels.py` が受け持つ。
FIGURES = (
    ("weight trajectories", weight_trajectories, "weight_trajectories.png"),
    ("weight distribution shift", weight_distribution_shift, "weight_distribution_shift.png"),
    ("firing rate scatter", firing_rate_scatter, "firing_rate_scatter.png"),
    ("mean correlation track", mean_correlation_track, "mean_correlation_track.png"),
    ("drive track", drive_track, "drive_track.png"),
    # 群ごとの同期。χ は metrics.csv の列、ヒートマップは各窓のスパイクから描く。
    ("chi track", chi_track, "chi_track.png"),
    ("participation heatmap", participation_heatmap, "participation_heatmap.png"),
    # 実効結合。**`rte.csv` を読む**ので、emit() の中で表を先に書く。
    ("rte track", rte_track, "rte_track.png"),
)


# 同じ段階で出す**報告の表**。形と回し方は `FIGURES` と同じ。今は無い。
TABLES = ()


def emit(series) -> None:
    """RTE を `data/` に記録し、run 全体の図と表を `<run>/report/overview/` へ。"""
    out = paths.report_dir(series.run_dir, paths.OVERVIEW)
    out.mkdir(parents=True, exist_ok=True)

    # --- 記録 (図より先) ---
    # **`FIGURES` に入れていない。** 置き場所が `data/` で、`rte track` の図がこれを読む。
    # 基準 (baseline の窓) が決まらないと値が決まらないので、1 行ずつ追記する
    # metrics.csv には混ぜない。baseline を持たない run では MissingData で skip される。
    guard("RTE table", write_rte_table, series, paths.data_dir(series.run_dir))

    for label, make, name in FIGURES + TABLES:
        guard(label, make, series, out / name)
