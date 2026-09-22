"""**build 直後**に出すもの。

シミュレーションの結果ではなく構築されたネットワークそのものを見るので、`setup()` の
前に出す (長い run が途中で落ちても構造の記録は残る)。再解析 (`replot.py`) は
**再ビルドして**同じものを描くため、この段階も再現できる。

**develop との違いはここ。** akita_soc は `area: no_space` の確率結合ネットワーク
(論文の N=100 再現) なので、空間を前提にした図 —— エリア・ネットワーク配置・軸索・
距離依存の結合確率 —— を持たない。持っていない図を並べて毎回 skip させるより、
出す側に置かない方が「何が出るか」が読める。空間を持つ発達実験は
`scripts/develop/report/structure.py` を見ること。
"""
from __future__ import annotations


from scripts.akita_soc.report import guard
from scripts.akita_soc.store import paths
from scripts.akita_soc.analysis.connectivity import write_report
from scripts.akita_soc.figures.synapse_hist import delay_distribution, weight_distribution
from scripts.akita_soc.figures.weight_matrix import weight_matrix


# **この段階で出るものの一覧。** 足すならここへ 1 行足す。
#
# `(ログ上の名前, 描く関数, ファイル名)`。置き場所は全部 `figures/structure/` なので
# 表には持たせない。`emit()` は**この表を上から順に回すだけ**。
FIGURES = (
    ("delay distribution", delay_distribution, "delay_distribution.png"),
    ("weight distribution", weight_distribution, "weight_distribution.png"),
    # 初期重みの行列。記録時刻ごとの同じ図は overview が出すので、**ここは 0 h の前**
    # (可塑性が 1 ステップも走っていない状態) を残すためにある。
    ("initial weight matrix", weight_matrix, "weight_matrix_initial.png"),
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
