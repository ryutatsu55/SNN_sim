"""重み行列の imshow。1 時点ぶんの 1 枚。

並べ替えは `ordering.py` の仕組みに乗る。COO を受け取り、描画の直前にだけ密へ戻す
(`densify`)。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scripts.akita_soc.figures import save, style

# 並べ替え軸。**ラスターや粗視化結合図とは別**で、ここは E/I だけでブロック化する
# (行列は N x N なので、module まで刻むと帯が細くなって読めない)。
ORDER_AXES = style.FALLBACK_ORDER_AXES

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


def weight_matrix(view, out_path: Path) -> None:
    """重み行列 1 枚。**build 直後 (`Built`) でも記録窓 (`Window`) でも同じ呼び出し。**

    `view.coo()` は「その時点の結合と重み」を返す契約なので、初期重みと記録時刻の重みを
    別の関数にする理由が無い。時刻を持つのは窓だけなので、タイトルだけ分ける。
    """
    coo = view.coo()
    layout = view.layout
    ordering = style.resolve_ordering(layout, ORDER_AXES)
    ordered = ordering.apply(densify(coo.row, coo.col, coo.weights, layout.total_neurons))

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    image = ax.imshow(ordered, origin="upper", interpolation="nearest",
                      vmin=VMIN, vmax=VMAX, cmap=CMAP)

    if ordering.enabled:
        style.draw_block_boundaries(ax, ordering, ordered.shape[0])
        ticks = style.block_ticks(ordering, ordered.shape[0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    hour = getattr(view, "hour", None)
    ax.set_title("Weight matrix (initial)" if hour is None
                 else f"Weight matrix {hour:g} h")
    ax.set_xlabel("post neuron id")
    ax.set_ylabel("pre neuron id")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="weight")
    save(fig, out_path, dpi=DPI)
