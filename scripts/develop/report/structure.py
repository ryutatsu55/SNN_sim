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


def emit(built) -> None:
    """構造図一式を `<run>/figures/structure/` へ、数値レポートを `data/` へ。"""
    run_dir = built.run_dir
    out = paths.fig_dir(run_dir, paths.STRUCTURE)
    out.mkdir(parents=True, exist_ok=True)

    print(f"  {built.total_neurons} neurons, {built.wiring().num_synapses} synapses")

    # --- 図 ---
    guard("area", area_figure, built, out / "area.png")
    guard("connection mask", connection_mask, built, out / "connection_mask_coarse.png")
    guard("delay distribution", delay_distribution, built, out / "delay_distribution.png")
    guard("weight distribution", weight_distribution, built, out / "weight_distribution.png")
    guard("network sample", network, built, out / "network_sample.png")
    guard("axon network", axon_network, built, out / "axon_network.png")
    # 「その距離のペアのうち何割が繋がったか」(確率) と「実際に張られた結合の長さが
    # 何本ずつか」(件数) の 2 枚。分母が違うので、片方が単調減少でももう片方はピークを持つ。
    guard("connection probability", empirical_connection_probability, built,
          out / "connection_probability.png")
    guard("distance distribution", distance_distribution, built,
          out / "distance_distribution.png")

    # --- 表 ---
    # 図で見えている濃淡が何倍の差なのかは絵からは読めないので、数値でも残す。
    guard("connectivity report", write_report, built, paths.data_dir(run_dir))

    print(f"Figures saved to: {out}")
