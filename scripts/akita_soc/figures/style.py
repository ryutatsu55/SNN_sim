"""この実験の図の約束。**2 枚以上の図が一致していなければならない値だけ**を置く。

置くのは 2 種類。

- **色** —— どの図でも EE が同じ色であること
- **並び順** —— ラスターの y 軸と行列の軸が同じ順であること (違うと 2 枚を並べて
  読む意味が消える)

1 枚だけが使う値 (その図の軸範囲・点の大きさ・figsize など) は、その図のファイル先頭に
置くこと —— ここへ集めると 1 枚の図を読むのに常に 2 ファイルを行き来することになる。

スタイルを**引数にしない**のがこの実験の方針。引数にすると、その関数が用意した枠の中
でしか見た目を変えられないうえ、変更のたびに呼び出し側も直すことになる。値を変えたく
なったら、その図のファイル (または、複数が共有するならここ) を直接書き換える。

**このモジュールは matplotlib を import しない。** 並べ替えの軸と順序は図だけのもの
ではなく、`analysis/connectivity.py` が出す群間結合確率の群分けでもある (絵と数値が
別の群で切られていたら突き合わせられない)。線を引く `draw_block_boundaries()` だけが
Axes を受け取るが、それも matplotlib の import を要さない。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 送信種別 × 受信種別の描画色。**3 枚 (重み行列・粗視化結合図・シナプス量ヒストグラム)
# が同じ色を使う。** どの図でも EE が同じ色であることが要点。
# ブロック名の正準順は `src.utils.analysis.weights.BLOCK_ORDER`。
BLOCK_COLORS = {
    "EE": "tab:red",
    "EI": "tab:orange",
    "IE": "tab:blue",
    "II": "tab:purple",
}

# 並べ替え。**この実験は E/I だけ。** akita_soc は `area: no_space` の確率結合
# ネットワーク (論文の N=100 再現) で、モジュール構造を持たないため。
# 空間を持つ発達実験は `scripts/develop/figures/style.py` で module を外側に置く。
ORDER_AXES = ("polarity",)
# 並べ替え軸が 1 つも使えないときの最後の拠り所。polarity は自動軸なので必ず存在する。
# 「E/I だけで並べればよい」図 (重み行列) もこれを直接指す。
FALLBACK_ORDER_AXES: tuple[str, ...] = ("polarity",)

# ブロック境界の線。**明背景 (粗視化結合図の白) でも暗背景 (重み行列の viridis 0) でも
# 読める中間グレー 1 色。** 白 1 色だと粗視化図で消え、黒 1 色だと重み行列で沈む。
BOUNDARY_COLOR = "0.5"
# 階層レベルごとの線幅 [pt]。外側 (レベル 0) が太い。レベルが増えても最後の値を使う。
BOUNDARY_WIDTHS = (1.0, 0.6)



def available_order_axes(layout, axes: tuple[str, ...] = ORDER_AXES):
    """layout が実際に持っている軸だけに絞った並べ替え軸を返す。

    無い軸を描画側へ渡すと `NetworkLayout` が KeyError を送出し、**記録時刻に到達した
    瞬間に長い run が落ちる**ので、ここで落としておく。

    「どの軸を使うか」を決めるのがここで、「その軸で実際に並べ替える」のが
    下の `resolve_ordering()`。
    """
    if layout is None or not axes:
        return None
    available = tuple(axis for axis in axes if layout.has_axis(axis))
    if available != tuple(axes):
        dropped = [axis for axis in axes if axis not in available]
        print(f"  Note: layout に無い並べ替え軸を除外しました: {dropped}")
    if not available:
        available = FALLBACK_ORDER_AXES
    return available


# ======================================================================================
# 並べ替えの実体
#
# 境界には**階層レベル**が付く。レベル 0 は最も外側の軸が切り替わる位置で、内側の軸だけが
# 切り替わる位置はレベル 1 以降。描画側はこれを見て線の太さを変え、入れ子を読めるようにする。
#
# `draw_block_boundaries()` がここにあるのは、重み行列と粗視化結合図の**2 枚が同じ境界線を
# 引く**から。片方に置くともう片方が借りに行くことになる。
# ======================================================================================



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

        「E/I は色で判別できるので線は引かない」という描画側の約束をここで表す。
        `boundaries` の level は `axes` の添字なので、軸名で素直に落とせる。
        境界そのものは `boundaries` に残るので、落とすのは描画対象からだけ。
        """
        return [(pos, level) for pos, level in self.boundaries if self.axes[level] not in skip]


def resolve_ordering(layout, order_axes: tuple[str, ...] | None = FALLBACK_ORDER_AXES) -> Ordering:
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


def draw_block_boundaries(ax, ordering: Ordering, size: int, *, scale: float = 1.0,
                           skip: tuple[str, ...] = ()) -> None:
    """ブロック境界に縦横の線を引く。外側の軸ほど太く描く。

    `scale` は「表示位置 (ニューロン単位) → 画像の画素」の倍率。1 ニューロン 1 画素の
    重み行列では 1.0、粗視化図では `grid / total_neurons` になる。

    `skip` に軸名を挙げるとその軸の切り替わりには線を引かない。E/I を**色**で示す図
    (粗視化図) は `skip=("polarity",)` を渡し、線の種類を 1 つに保つ。値そのものを色に
    使っている図 (重み行列) は既定の `()` のままで、E/I 境界も線で示す。

    **線は 1 本 1 色。** 以前は白の上に細い黒を重ねていたが、黒が白の中央を削るので
    「白 1px + 黒 1px + 白 1px」になり、画素内の位置によって太さも濃さもばらついた。
    """
    for position, level in ordering.visible_boundaries(skip):
        position = position * scale
        if not 0 < position < size:
            continue
        offset = position - 0.5
        width = BOUNDARY_WIDTHS[min(level, len(BOUNDARY_WIDTHS) - 1)]
        for line in (ax.axhline(offset, color=BOUNDARY_COLOR, linewidth=width),
                     ax.axvline(offset, color=BOUNDARY_COLOR, linewidth=width)):
            # **アンチエイリアスを切り、画素へスナップさせる。** 境界位置は整数でも、
            # データ座標から画素への倍率が整数でないので線ごとに画素内の位置がずれる。
            # そのまま描くと 1 本ずつ滲み方が変わり、太さも濃さもばらつく。
            line.set_antialiased(False)
            line.set_snap(True)
