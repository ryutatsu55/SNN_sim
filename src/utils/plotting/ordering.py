"""表示のための並べ替え軸。

`NetworkLayout` の軸を使って「どの順にニューロンを並べるか」を決め、ブロック境界の
位置まで求める。重み行列の imshow もラスターの y 軸も同じ結果を使うので、
「層でブロック化し、各層の中で興奮性→抑制性」のような指定が両方に同じ形で効く。

    order_axes=("polarity",)          既定。興奮性を先頭ブロックに置く
    order_axes=("layer", "polarity")  層でブロック化し、層内で E→I
    order_axes=None                   並べ替えない (グローバル ID の順)

境界には**階層レベル**が付く。レベル 0 は最も外側の軸 (上の例なら `layer`) が切り替わる
位置で、内側の軸だけが切り替わる位置はレベル 1 以降になる。描画側はこれを見て線の太さを
変え、ブロックの入れ子を読めるようにする。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 既定の並べ替え。興奮性 (excitatory) が抑制性 (inhibitory) より前に来る。
DEFAULT_ORDER_AXES: tuple[str, ...] = ("polarity",)


@dataclass(frozen=True)
class Ordering:
    """並べ替えの結果。`axes` が空なら並べ替えなしを表す。"""
    axes: tuple[str, ...] = ()
    order: np.ndarray | None = None   # 表示位置 -> グローバルID
    rank: np.ndarray | None = None    # グローバルID -> 表示位置
    boundaries: list[tuple[int, int]] = field(default_factory=list)  # (表示位置, レベル)

    @property
    def enabled(self) -> bool:
        return self.order is not None

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        """(N, N) 行列の行と列を同じ順に並べ替える。"""
        if not self.enabled:
            return matrix
        return matrix[np.ix_(self.order, self.order)]

    def positions(self, level: int | None = None) -> list[int]:
        """境界の表示位置。`level` を指定するとその階層のものだけ返す。"""
        return [pos for pos, lv in self.boundaries if level is None or lv == level]


def resolve_ordering(layout, order_axes: tuple[str, ...] | None = DEFAULT_ORDER_AXES) -> Ordering:
    """並べ替え順とブロック境界を求める。

    Args:
        layout: NetworkLayout。None なら並べ替えなし。
        order_axes: 並べ替えに使う軸を外側から順に。None または空なら並べ替えなし。

    軸が layout に無ければ `NetworkLayout` 側が送出する例外がそのまま伝わる
    (`layer` 軸を持たない run に `order_axes=("layer",)` を指定した場合など)。
    """
    if layout is None or not order_axes:
        return Ordering()

    axes = tuple(order_axes)
    order = layout.order_by(*axes)
    total = order.size

    # 並べ替え後に軸値が変わる位置を求める。外側の軸が変われば内側も必ず変わるので、
    # 「最初に変化した軸」がその境界の階層レベルになる。
    level_of_change = np.full(max(total - 1, 0), -1, dtype=np.int64)
    for level, axis in enumerate(axes):
        values = layout.labels(axis)[order]
        changed = values[1:] != values[:-1]
        level_of_change[(level_of_change < 0) & changed] = level

    boundaries = [(int(i) + 1, int(level_of_change[i]))
                  for i in np.nonzero(level_of_change >= 0)[0]]

    rank = np.empty(total, dtype=np.int64)
    rank[order] = np.arange(total, dtype=np.int64)
    return Ordering(axes=axes, order=order, rank=rank, boundaries=boundaries)


def block_ticks(ordering: Ordering, size: int) -> list[int]:
    """最外ブロックの区切りが読める目盛り位置を返す (先頭・各ブロックの末尾・最終)。"""
    ticks = [0]
    ticks.extend(pos - 1 for pos in ordering.positions(level=0))
    ticks.append(size - 1)
    return sorted({tick for tick in ticks if 0 <= tick < size})
