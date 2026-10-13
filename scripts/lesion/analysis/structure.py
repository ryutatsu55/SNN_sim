"""配線のトポロジー指標を、切断の前と後で 1 回ずつ測って `structure.csv` に書く。

**結合の有無だけで決まる量**がここに来る —— 次数・連結成分・participation と
ハブの数・媒介中心性・クラスタ係数。切断後の配線は run を通して固定 (可塑性が変えるのは
重みだけ) なので、probe ごとに測っても同じ値が並ぶだけになる。重みを使う量
(strength・E/I ブロックの重み統計) は probe ごとに変わるので `metrics.csv` の担当。

**解析のパラメータ (ハブ判定の閾値・媒介中心性の厳密解の上限など) は渡さない。**
それを使う関数 (`src/utils/analysis/graph.py`) が既定値として持っているので、
ここで上書きすると定義が 2 か所になる。
"""
from __future__ import annotations

from pathlib import Path

from src.utils.analysis.graph import graph_metrics

from scripts.lesion.store.formats import write_table

MODULE_AXIS = "module"


def structure_rows(built) -> list[dict]:
    """切断前 (`wiring = "pre"`) と切断後 (`"post"`) の 2 行を返す。"""
    layout = built.layout
    labels = layout.labels(MODULE_AXIS) if layout.has_axis(MODULE_AXIS) else None
    rows = []
    for name, wiring in (("pre", built.wiring_pre()), ("post", built.wiring())):
        row = {"wiring": name}
        row.update(graph_metrics(wiring.row, wiring.col, built.total_neurons, labels=labels))
        rows.append(row)
    return rows


def write_structure(built, out_path: Path) -> None:
    """`structure_rows()` を `structure.csv` へ書く。"""
    write_table(structure_rows(built), out_path)
