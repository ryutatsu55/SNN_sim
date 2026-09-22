"""構築されたネットワークの結合確率を測る。

粗視化結合図 (`figures/connection_mask.py`) が絵で見せている濃淡を、**数値でも
残す**。図からは「何倍の差か」が読めないため。

表になるものは CSV にして `data/` へ置き、run.log には人間向けに整形した同じ内容を出す。
プールした within / between / segregation の 3 値は CSV の `synapses` と `pairs` から
再計算できる派生値なので、列には持たせず run.log だけに出す。

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

from src.utils.runview import optional
from scripts.akita_soc.store.records import write_table
# 群分けの軸は**粗視化図と同じもの**でなければ、絵と数値が別のことを言う。選び方を
# 2 つ持たないよう `style` から借りる (`style.py` は matplotlib を持たないので、
# 「analysis は matplotlib を import しない」は保たれる)。
from scripts.akita_soc.figures import style

PROBABILITY_NAME = "connection_probability.csv"
HOPS_NAME = "bridge_hops.csv"


def write_report(built, out_dir: Path) -> None:
    """群間結合確率を CSV に書き、整形した表を標準出力 (= run.log) へ出す。

    軸は粗視化図と同じ最外軸。並べ替え軸を持たない run では何も出さない。
    `out_dir` は CSV の置き場所 (= `data/`)。2 本まとめて出すのでファイル名はここが決める。
    """
    order_axes = style.available_order_axes(built.layout)
    if not order_axes:
        return
    wiring = built.wiring()
    result = group_connection_probability(wiring.row, wiring.col, built.layout, order_axes[0])

    out_dir = Path(out_dir)
    write_table(_probability_rows(result), out_dir / PROBABILITY_NAME)
    print()
    print(format_group_connection_probability(result))
    print()

    hops = _hop_matrix(result, optional(built.area))
    if hops is None:
        return
    by_hops = hop_connection_probability(result, hops)
    write_table(_hop_rows(by_hops), out_dir / HOPS_NAME)
    print(format_hop_connection_probability(by_hops))
    print()


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


def _hop_matrix(result, area):
    """ブリッジのホップ数行列。エリアがブリッジを持たない構成なら理由を 1 行出して None。"""
    if area is None:
        return None
    try:
        return bridge_hop_matrix(area, result.names)
    except ValueError as exc:
        print(f"  ブリッジのホップ数別の集計はスキップしました: {exc}")
        print()
        return None
