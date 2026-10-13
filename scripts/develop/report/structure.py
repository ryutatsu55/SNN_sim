"""**build 直後**に出すもの。

シミュレーションの結果ではなく構築されたネットワークそのものを見るので、`setup()` の
前に出す (長い run が途中で落ちても構造の記録は残る)。受け取るのは `series.built()`
(build の記録を読む view) で、再解析 (`replot.py`) も同じものを渡すので、この段階も再現できる。

図が 1 枚描けなくても止めない。`no_space` の config には座標が無く、`constant_prob` は
軸索の幾何を持たないが、それは異常ではないので `MissingData` として 1 枚ずつ報告される。
"""
from __future__ import annotations


from scripts.develop.report import guard
from scripts.develop.store import paths
from scripts.develop.analysis.connectivity import (bridge_hops_table,
                                                   connection_probability_table)
from scripts.develop.figures.area import area_figure
from scripts.develop.figures.connection_mask import (connection_mask,
                                                     empirical_connection_probability)
from scripts.develop.figures.network import axon_network, network
from scripts.develop.figures.synapse_hist import (delay_distribution, distance_distribution,
                                                  weight_distribution)


# **この段階で出るものの一覧。** 足すならここへ 1 行足す。
#
# `(ログ上の名前, 描く関数, ファイル名)`。置き場所は全部 `report/structure/` なので
# 表には持たせない。`emit()` は**この表を上から順に回すだけ**。
FIGURES = (
    ("area", area_figure, "area.png"),
    ("connection mask", connection_mask, "connection_mask_coarse.png"),
    ("delay distribution", delay_distribution, "delay_distribution.png"),
    # build 直後 = **可塑性が 1 ステップも走る前**の重み。記録時刻ごとの同じ図は
    # panels が出すので、ここは 0 h の手前を残すためにある。
    ("initial weight distribution", weight_distribution,
     "weight_distribution_initial.png"),
    ("network sample", network, "network_sample.png"),
    ("axon network", axon_network, "axon_network.png"),
    # 「その距離のペアのうち何割が繋がったか」(確率) と「実際に張られた結合の長さが
    # 何本ずつか」(件数) の 2 枚。分母が違うので、片方が単調減少でももう片方はピークを持つ。
    ("connection probability", empirical_connection_probability,
     "connection_probability.png"),
    ("distance distribution", distance_distribution, "distance_distribution.png"),
)


# 同じ段階で出す**報告の表**。形と回し方は `FIGURES` と同じで、置き場所も同じ
# `report/structure/`。図で見えている濃淡が何倍の差なのかは絵からは読めないので、数値でも残す。
TABLES = (
    ("connection probability table", connection_probability_table,
     "connection_probability.csv"),
    # ブリッジを持たない area では MissingData で skip される。
    ("bridge hops table", bridge_hops_table, "bridge_hops.csv"),
)


def emit(built) -> None:
    """構造の図と表を `<run>/report/structure/` へ。"""
    run_dir = built.run_dir
    out = paths.report_dir(run_dir, paths.STRUCTURE)
    out.mkdir(parents=True, exist_ok=True)

    print(f"  {built.total_neurons} neurons, {built.wiring().num_synapses} synapses")

    for label, make, name in FIGURES + TABLES:
        guard(label, make, built, out / name)

    print(f"Report saved to: {out}")
