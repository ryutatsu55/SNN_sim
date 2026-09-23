"""重み行列の imshow。probe 1 点ぶんの 1 枚。

並べ替えは `style.py` の仕組みに乗り、**粗視化結合図と同じ module > polarity** で
ブロック化する。COO を受け取り、描画の直前にだけ密へ戻す (`densify`)。

色は重みの値そのもの (viridis)。E/I ブロックの色分けは持たない —— 色を 2 つの意味に
使えないので、群の区別は**並び順**が受け持つ。線は最外ブロック (module) にだけ引く。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scripts.lesion.figures import save, style

# 並べ替え軸。**粗視化結合図 (connection_mask) と同じ** module > polarity。
# 2 枚は同じブロック配置になるので、「どの群がどれだけ繋がっているか」(結合図) と
# 「その群の重みがいくつか」(この図) を位置で突き合わせられる。
ORDER_AXES = style.ORDER_AXES

# 重み行列の見た目。**引数にしない** —— 変えたくなったらここを直す。
SINGLE_FIGSIZE = (6, 5.4)
VMIN, VMAX = 0.0, 1.0         # 重みは [0, 1] (可塑性の Wmax で正規化済み)
CMAP = "viridis"
DPI = 200

DENSE_RENDER_LIMIT = 20000
# 最外ブロックの名前を目盛りに出す上限。これを超えると文字が潰れるので番号のままにする。
MAX_BLOCK_TICKS = 16


def densify(row: np.ndarray, col: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """COO を imshow 用の (size, size) 行列へ起こす。**描画直前のローカルな密化**。

    結合が無い箇所は 0 で埋まる。これは「絵として黒い」だけで統計には使わないこと
    (統計は COO のまま `analysis.weights.block_values` で取る)。
    """
    size = int(size)
    if size > DENSE_RENDER_LIMIT:
        raise MemoryError(
            f"N={size} の重み行列を画像にするには密な {size**2 * 8 / 2**30:.1f} GiB が"
            f" 必要です (上限 N={DENSE_RENDER_LIMIT})。"
            " 粗視化図 (connection_mask) を使ってください。"
        )
    values = np.asarray(values).reshape(-1)
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    if not (row.size == col.size == values.size):
        raise ValueError("row / col / values の長さが一致しません。")
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[row, col] = values
    return matrix


def weight_matrix(window, out_path: Path) -> None:
    """probe 1 点ぶんの重み行列を可視化する。

    **切断前の probe は切断前の結合、切断後は切断後の結合**を描く (`window.wiring()`
    が phase で切り替える)。したがって baseline と post では行列に載る本数が違う。
    """
    coo = window.coo()
    layout = window.layout
    ordering = style.resolve_ordering(layout, style.available_order_axes(layout, ORDER_AXES))
    ordered = ordering.apply(densify(coo.row, coo.col, coo.weights, layout.total_neurons))

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    image = ax.imshow(ordered, origin="upper", interpolation="nearest",
                      vmin=VMIN, vmax=VMAX, cmap=CMAP)

    if ordering.enabled:
        # 線は**最外ブロック (module) の境界だけ**。粗視化結合図と同じ約束で、
        # E/I の切り替わりには引かない —— 2 種類の線が混ざると格子が読めなくなる。
        style.draw_block_boundaries(ax, ordering, ordered.shape[0], skip=("polarity",))
        # 最外ブロックの名前を目盛りに。**粗視化結合図と同じ名前が同じ位置に出る。**
        # 多すぎると文字が潰れるので、その場合だけニューロン番号に落とす
        # (module が 16 個あると番号同士が重なって読めなくなるため)。
        blocks = style.outer_block_labels(layout, ordering, ordered.shape[0])
        if blocks and len(blocks) <= MAX_BLOCK_TICKS:
            ticks = [centre - 0.5 for centre, _ in blocks]
            names = [name for _, name in blocks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_yticks(ticks)
            ax.set_yticklabels(names)
        else:
            ticks = style.block_ticks(ordering, ordered.shape[0])
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

    ax.set_title(f"Weight matrix {window.label}")
    ax.set_xlabel("post neuron id")
    ax.set_ylabel("pre neuron id")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="weight")
    save(fig, out_path, dpi=DPI)
