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
# polarity は自動軸なので**どの layout にも必ずある**。最後の拠り所として使える。
DEFAULT_ORDER_AXES: tuple[str, ...] = ("polarity",)

# モジュール構造を持つ run で使いたい並べ替え。module でブロック化し、各モジュール内で E→I。
# **ラスターと粗視化結合図が同じ並びになる**ようにここで 1 つに決める。
# module 軸は `space: area_uniform` + 複合エリアの run にしか無いので、
# `available_order_axes()` が layout を見て落とす。
GROUPED_ORDER_AXES: tuple[str, ...] = ("module", "polarity")


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

    def visible_boundaries(self, skip: tuple[str, ...] = ("polarity",)) -> list[tuple[int, int]]:
        """`skip` に挙げた軸の切り替わり位置を除いた境界。

        落とすのは描画対象からだけで、境界そのものは `boundaries` に残る。
        """
        return [(pos, level) for pos, level in self.boundaries if self.axes[level] not in skip]


def resolve_ordering(layout, order_axes: tuple[str, ...] | None = DEFAULT_ORDER_AXES) -> Ordering:
    """並べ替え順とブロック境界を求める。

    Args:
        layout: NetworkLayout。None なら並べ替えなし。
        order_axes: 並べ替えに使う軸を外側から順に。None または空なら並べ替えなし。
            layout が持たない軸を渡すと `NetworkLayout` の例外がそのまま伝わる。
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


def available_order_axes(layout, axes: tuple[str, ...] = GROUPED_ORDER_AXES):
    """layout が実際に持っている軸だけに絞った並べ替え軸を返す。

    **「どの軸を使うか」を決めるのがここ**で、「その軸で並べ替える」のが
    `resolve_ordering()`。持たない軸を落とすので、run の途中で例外にならない。
    """
    if layout is None or not axes:
        return None
    available = tuple(axis for axis in axes if layout.has_axis(axis))
    if available != tuple(axes):
        dropped = [axis for axis in axes if axis not in available]
        print(f"  Note: layout に無い並べ替え軸を除外しました: {dropped}")
    if not available:
        available = DEFAULT_ORDER_AXES
    return available


def draw_block_boundaries(ax, ordering: Ordering, size: int, *, scale: float = 1.0,
                          skip: tuple[str, ...] = ()) -> None:
    """ブロック境界に縦横の線を引く。外側の軸ほど太く描く。

    Args:
        scale: 表示位置 (ニューロン単位) → 画像の画素の倍率。1 ニューロン 1 画素の
            重み行列では 1.0、粗視化図では `grid / total_neurons`。
        skip: 挙げた軸の切り替わりには線を引かない。E/I を色で示す図に使う。
    """
    for position, level in ordering.visible_boundaries(skip):
        position = position * scale
        if not 0 < position < size:
            continue
        # 白線の上に細い黒線を重ねると、明背景でも暗背景でも見える。
        # 縦横それぞれ白 → 黒の順に引くこと (交点で黒が上に来る)。
        white, black = (1.2, 0.4) if level == 0 else (0.8, 0.25)
        offset = position - 0.5
        ax.axhline(offset, color="white", linewidth=white)
        ax.axvline(offset, color="white", linewidth=white)
        ax.axhline(offset, color="black", linewidth=black)
        ax.axvline(offset, color="black", linewidth=black)
