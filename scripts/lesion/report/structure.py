"""**切断の直後**に出すもの。

develop の `structure.py` が「build 直後」に出すのに対し、lesion は**切断が済んだ
ネットワーク**の形を出す。切断前の形は親 run の `report/structure/` に残っているので、
ここで両方は出さない。例外は記録の `structure.csv` で、切断の前後を 1 行ずつ並べる。

受け取るのは `series.built()` (切断直後の記録を読む view)。軸索幾何は切断後に絞った
ものが記録されている (`store/built.py`) ので、切ったはずの結合は描かれない。

図が 1 枚描けなくても止めない。`no_space` の config には座標が無く、`constant_prob` は
軸索の幾何を持たないが、それは異常ではないので `MissingData` として 1 枚ずつ報告される。
"""
from __future__ import annotations


from scripts.lesion.report import guard
from scripts.lesion.store import paths
from scripts.lesion.analysis.connectivity import (bridge_hops_table,
                                                  connection_probability_table)
from scripts.lesion.analysis.metrics import cut_profile_table
from scripts.lesion.analysis.structure import write_structure
from scripts.lesion.store.formats import STRUCTURE_NAME
from scripts.lesion.figures.area import area_figure
from scripts.lesion.figures.connection_mask import (connection_mask,
                                                    empirical_connection_probability)
from scripts.lesion.figures.network import axon_network, network
from scripts.lesion.figures.synapse_hist import (delay_distribution, distance_distribution,
                                                 weight_distribution)


# **この段階で出るものの一覧。** 足すならここへ 1 行足す。
#
# `(ログ上の名前, 描く関数, ファイル名)`。置き場所は全部 `report/structure/` なので
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


# 同じ段階で出す**報告の表**。形と回し方は `FIGURES` と同じで、置き場所も同じ
# `report/structure/`。切断で群間結合がどれだけ減ったかは、この表を親 run のものと比べれば読める。
TABLES = (
    ("connection probability table", connection_probability_table,
     "connection_probability.csv"),
    # ブリッジを持たない area では MissingData で skip される。
    ("bridge hops table", bridge_hops_table, "bridge_hops.csv"),
    # 切ったシナプスと残したシナプスの属性の比較。「何を失ったか」を残存群と並べて読む。
    ("cut profile table", cut_profile_table, "cut_profile.csv"),
)


def emit(built) -> None:
    """切断前後のトポロジーを `data/` に記録し、構造の図と表を `<run>/report/structure/` へ。"""
    run_dir = built.run_dir

    # --- 記録 ---
    # **`guard()` を通さない。** 落ちたら記録が欠けるので、警告 1 行で流さずに止める。
    write_structure(built, paths.data_path(run_dir, STRUCTURE_NAME))

    out = paths.report_dir(run_dir, paths.STRUCTURE)
    out.mkdir(parents=True, exist_ok=True)

    print(f"  切断後: {built.total_neurons} neurons, {built.wiring().num_synapses} synapses")

    for label, make, name in FIGURES + TABLES:
        guard(label, make, built, out / name)

    print(f"Report saved to: {out}")
