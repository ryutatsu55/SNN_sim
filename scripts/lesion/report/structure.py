"""**切断の直後**に出すもの。

develop の `structure.py` が「build 直後」に出すのに対し、lesion は**切断が済んだ
ネットワーク**の形を出す。切断前の形は親 run の `figures/structure/` に残っているので、
ここで両方は出さない。

受け取るのは `series.built()` (切断直後の記録を読む view)。軸索幾何は切断後に絞った
ものが記録されている (`store/built.py`) ので、切ったはずの結合は描かれない。

図が 1 枚描けなくても止めない。`no_space` の config には座標が無く、`constant_prob` は
軸索の幾何を持たないが、それは異常ではないので `MissingData` として 1 枚ずつ報告される。
"""
from __future__ import annotations


from scripts.lesion.report import guard
from scripts.lesion.store import paths
from scripts.lesion.analysis.connectivity import write_report
from scripts.lesion.figures.area import area_figure
from scripts.lesion.figures.connection_mask import (connection_mask,
                                                    empirical_connection_probability)
from scripts.lesion.figures.network import axon_network, network
from scripts.lesion.figures.synapse_hist import (delay_distribution, distance_distribution,
                                                 weight_distribution)


# **この段階で出るものの一覧。** 足すならここへ 1 行足す。
#
# `(ログ上の名前, 描く関数, ファイル名)`。置き場所は全部 `figures/structure/` なので
# 表には持たせない。`emit()` は**この表を上から順に回すだけ**。
FIGURES = (
    ("area", area_figure, "area.png"),
    ("connection mask", connection_mask, "connection_mask_coarse.png"),
    ("delay distribution", delay_distribution, "delay_distribution.png"),
    # 切断直後 = **切断後の可塑性が 1 ステップも走る前**の重み。probe ごとの同じ図は
    # panels が出すので、ここは 0 h の手前を残すためにある。
    # **名前は panels のものと変えること** —— ログの `skip X` がどちらの話か分からなくなる。
    ("weight distribution at cut", weight_distribution, "weight_distribution.png"),
    ("network sample", network, "network_sample.png"),
    ("axon network", axon_network, "axon_network.png"),
    # 「その距離のペアのうち何割が繋がったか」(確率) と「実際に張られた結合の長さが
    # 何本ずつか」(件数) の 2 枚。分母が違うので、片方が単調減少でももう片方はピークを持つ。
    ("connection probability", empirical_connection_probability,
     "connection_probability.png"),
    ("distance distribution", distance_distribution, "distance_distribution.png"),
)


def emit(built) -> None:
    """構造図一式を `<run>/figures/structure/` へ、数値レポートを `data/` へ。"""
    run_dir = built.run_dir
    out = paths.fig_dir(run_dir, paths.STRUCTURE)
    out.mkdir(parents=True, exist_ok=True)

    print(f"  切断後: {built.total_neurons} neurons, {built.wiring().num_synapses} synapses")

    for label, draw, name in FIGURES:
        guard(label, draw, built, out / name)

    # --- 表 ---
    # **`FIGURES` に入れていない。** 置き場所が `figures/` ではなく `data/` で、
    # 渡すのもファイル名ではなくディレクトリだから (2 本まとめて出す)。
    # 切断で群間結合がどれだけ減ったかは、この表を親 run のものと比べれば読める。
    guard("connectivity report", write_report, built, paths.data_dir(run_dir))

    print(f"Figures saved to: {out}")
