"""構築されたネットワークの群間結合確率を、報告の表にする。

粗視化結合図 (`figures/connection_mask.py`) が絵で見せている濃淡を、**数値でも
残す**。図からは「何倍の差か」が読めないため。

どちらの関数も図と同じ `(view, out_path)` を取り、表を CSV に書いて、人向けに整形した
同じ内容を標準出力 (= run.log) へ出す。プールした within / between / segregation の
3 値は CSV の `synapses` と `pairs` から再計算できる派生値なので、列には持たせず
run.log だけに出す。

「どう測るか」(結合確率の定義・ブリッジのホップ数) は `src/utils/analysis/connectivity.py`。
ここが決めるのは**何を測り、どの列で残すか**だけ。
"""
from __future__ import annotations

from pathlib import Path

from src.utils.analysis.connectivity import (
    bridge_hop_matrix,
    format_group_connection_probability,
    format_hop_connection_probability,
    group_connection_probability,
    hop_connection_probability,
)

from src.utils.runview import MissingData
from scripts.lesion.store.formats import write_table
# 群分けの軸は**粗視化図と同じもの**でなければ、絵と数値が別のことを言う。選び方を
# 2 つ持たないよう `style` から借りる (`style.py` は matplotlib を持たないので、
# 「analysis は matplotlib を import しない」は保たれる)。
from scripts.lesion.figures import style


def connection_probability_table(built, out_path: Path) -> None:
    """群間の結合確率を 1 ペア 1 行の表にして書き、整形した表を標準出力へ出す。

    群分けは粗視化図と同じ最外軸。並べ替え軸を持たない run では `MissingData`。
    """
    result = _group_probability(built)
    write_table(_probability_rows(result), out_path)
    print()
    print(format_group_connection_probability(result))
    print()


def bridge_hops_table(built, out_path: Path) -> None:
    """群間の結合確率をブリッジのホップ数ごとに集計して書き、整形した表を標準出力へ出す。

    エリアがブリッジを持たない構成では `MissingData`。
    """
    result = _group_probability(built)
    area = built.area()
    try:
        hops = bridge_hop_matrix(area, result.names)
    except ValueError as exc:
        raise MissingData("bridges", str(exc)) from exc
    by_hops = hop_connection_probability(result, hops)
    write_table(_hop_rows(by_hops), out_path)
    print(format_hop_connection_probability(by_hops))
    print()


def _group_probability(built):
    order_axes = style.available_order_axes(built.layout)
    if not order_axes:
        raise MissingData("order axes", "群分けに使う並べ替え軸を layout が持ちません")
    wiring = built.wiring()
    return group_connection_probability(wiring.row, wiring.col, built.layout, order_axes[0])


def _probability_rows(result) -> list[dict]:
    """K×K の表を 1 ペア 1 行の long 形式にする (CSV は縦持ちの方が扱いやすい)。"""
    return [
        {
            "source": source,
            "target": target,
            "synapses": int(result.counts[i, j]),
            "pairs": int(result.possible[i, j]),
            "probability": float(result.probability[i, j]),
        }
        for i, source in enumerate(result.names)
        for j, target in enumerate(result.names)
    ]


def _hop_rows(result) -> list[dict]:
    """ホップ数ごとに 1 行。`hops` の -1 は到達不能 (連結していない) を表す。"""
    return [
        {
            "hops": int(level),
            "group_pairs": int(result.group_pairs[i]),
            "pairs": int(result.possible[i]),
            "synapses": int(result.counts[i]),
            "probability": float(result.probability[i]),
        }
        for i, level in enumerate(result.levels)
    ]
