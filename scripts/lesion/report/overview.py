"""**run 終了後**に出すもの。全 probe をまとめて見る図。

本番の run の末尾も、再解析もここを通る。1 枚描けなくても止めない —— ここへ来る時点で
npz も `metrics.csv` も確定しているので、**回復過程を丸ごと回した結果を描画の都合で
失わない**。
"""
from __future__ import annotations

from scripts.lesion.report import guard
from scripts.lesion.store import paths
from scripts.lesion.analysis.effective import write_rte_table
from scripts.lesion.analysis.flip_rule import write_flip_timing
from scripts.lesion.analysis.input_budget import write_input_budget, write_neuron_budget
from scripts.lesion.figures.change import (correlation_change_matrix,
                                           correlation_change_matrix_across_cut,
                                           correlation_change_network,
                                           correlation_change_network_across_cut,
                                           post_rate_vs_weight_change,
                                           pre_rate_vs_weight_change,
                                           weight_change_axon_network,
                                           weight_change_axon_network_across_cut,
                                           weight_change_matrix,
                                           weight_change_matrix_across_cut,
                                           weight_change_network,
                                           weight_change_network_across_cut,
                                           weight_change_vs_correlation,
                                           weight_change_vs_correlation_change)
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
    # 切断直後から `change.TARGET_HOUR` (20 h) までの変化 (回復)。
    # その時刻の窓を持たない run では MissingData で skip される。
    ("weight change matrix", weight_change_matrix, "weight_change_matrix.png"),
    ("weight change network", weight_change_network, "weight_change_network.png"),
    ("weight change axon network", weight_change_axon_network,
     "weight_change_axon_network.png"),
    ("correlation change matrix", correlation_change_matrix,
     "correlation_change_matrix.png"),
    ("correlation change network", correlation_change_network,
     "correlation_change_network.png"),
    # 切断前から切断直後までの変化 (切断そのものの効果)。
    ("weight change matrix across cut", weight_change_matrix_across_cut,
     "weight_change_matrix_across_cut.png"),
    ("weight change network across cut", weight_change_network_across_cut,
     "weight_change_network_across_cut.png"),
    ("weight change axon network across cut", weight_change_axon_network_across_cut,
     "weight_change_axon_network_across_cut.png"),
    ("correlation change matrix across cut", correlation_change_matrix_across_cut,
     "correlation_change_matrix_across_cut.png"),
    ("correlation change network across cut", correlation_change_network_across_cut,
     "correlation_change_network_across_cut.png"),
    # 切断後に残った結合 1 本 = 1 点。回復の重みの差を、相関や 20 h の発火率と並べる。
    ("weight change vs correlation change", weight_change_vs_correlation_change,
     "weight_change_vs_correlation_change.png"),
    ("weight change vs correlation", weight_change_vs_correlation,
     "weight_change_vs_correlation.png"),
    ("pre rate vs weight change", pre_rate_vs_weight_change, "pre_rate_vs_weight_change.png"),
    ("post rate vs weight change", post_rate_vs_weight_change, "post_rate_vs_weight_change.png"),
)


# 同じ段階で出す**報告の表**。形と回し方は `FIGURES` と同じ。
TABLES = (
    # 強化された E 発の結合ごとに、D が正になった時刻と重みが 0.5 を超えた時刻 (どちらが先か)。
    ("flip timing", write_flip_timing, "flip_timing.csv"),
    # 駆動の収支。切断で失った駆動と 20 h までに得た駆動を、結合ごと / ニューロンごとに分解する。
    ("input budget", write_input_budget, "input_budget.csv"),
    ("neuron budget", write_neuron_budget, "neuron_budget.csv"),
)


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
