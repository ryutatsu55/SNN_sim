"""**build 直後**に出すもの。

シミュレーションの結果ではなく構築されたネットワークそのものを見るので、`setup()` の
前に出す (長い run が途中で落ちても構造の記録は残る)。再解析 (`replot.py`) は
**再ビルドして**同じものを描くため、この段階も再現できる。

図が 1 枚描けなくても止めない。`no_space` の config には座標が無く、`constant_prob` は
軸索の幾何を持たないが、それは異常ではないので `MissingData` として 1 枚ずつ報告される。
"""
from __future__ import annotations


from scripts.develop.report import guard
from scripts.develop.store import paths
from scripts.develop.analysis.connectivity import write_report
from scripts.develop.figures.area import area_figure
from scripts.develop.figures.connection_mask import (connection_mask,
                                                     empirical_connection_probability)
from scripts.develop.figures.network import axon_network, network
from scripts.develop.figures.synapse_hist import (delay_distribution, distance_distribution,
                                                  weight_distribution)


# **この段階で出るものの一覧。** 足すならここへ 1 行足す。
#
# `(ログ上の名前, 描く関数, ファイル名)`。置き場所は全部 `figures/structure/` なので
# 表には持たせない。`emit()` は**この表を上から順に回すだけ**。
FIGURES = (
    ("area", area_figure, "area.png"),
    ("connection mask", connection_mask, "connection_mask_coarse.png"),
    ("delay distribution", delay_distribution, "delay_distribution.png"),
    ("weight distribution", weight_distribution, "weight_distribution.png"),
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

    print(f"  {built.total_neurons} neurons, {built.wiring().num_synapses} synapses")

    for label, draw, name in FIGURES:
        guard(label, draw, built, out / name)

    # --- 表 ---
    # **`FIGURES` に入れていない。** 置き場所が `figures/` ではなく `data/` で、
    # 渡すのもファイル名ではなくディレクトリだから (2 本まとめて出す)。
    # 図で見えている濃淡が何倍の差なのかは絵からは読めないので、数値でも残す。
    guard("connectivity report", write_report, built, paths.data_dir(run_dir))

    print(f"Figures saved to: {out}")
