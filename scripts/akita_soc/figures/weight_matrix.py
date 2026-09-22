"""重み行列の imshow。1 時刻ぶんの 1 枚と、時系列を並べたパネル。

並べ替えは `ordering.py` の仕組みに乗る。COO を受け取り、描画の直前にだけ密へ戻す
(`densify`)。
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import matplotlib.pyplot as plt
import numpy as np
from scripts.akita_soc.figures import save, style

# 並べ替え軸。**ラスターや粗視化結合図とは別**で、ここは E/I だけでブロック化する
# (行列は N x N なので、module まで刻むと帯が細くなって読めない)。
ORDER_AXES = style.FALLBACK_ORDER_AXES

# 重み行列の見た目。**引数にしない** —— 変えたくなったらここを直す。
SINGLE_FIGSIZE = (6, 5.4)
PANEL_CELL = (4.2, 3.9)       # パネル 1 コマぶんの (幅, 高さ)
VMIN, VMAX = 0.0, 1.0         # 重みは [0, 1] (可塑性の Wmax で正規化済み)
CMAP = "viridis"
# 差分パネルは値域が [-1, 1] なので発散配色。
DELTA_VMIN, DELTA_VMAX = -1.0, 1.0
DELTA_CMAP = "coolwarm"
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


def weight_panel(series, out_path: Path) -> None:
    """記録時刻ごとの重み行列を時系列に並べる。"""
    items = [(window.hour, window.weights()) for window in series.windows]
    _panel(series, items, out_path,
           title=f"Weight matrix timeline: {series.run_dir.name}",
           vmin=VMIN, vmax=VMAX, cmap=CMAP)


def weight_delta_panel(series, out_path: Path) -> None:
    """前の記録時刻からの**変化量**を時系列に並べる。

    重みそのものの図と**別の関数にしてある。** 値域が [0, 1] ではなく [-1, 1] で、
    発散配色でなければ増減が読めない。引数で切り替える形にすると、この 2 枚が
    「カラーマップ以外は同じでなければならない」という制約に縛られる。
    """
    _panel(series, list(_weight_deltas(series)), out_path,
           title=f"Weight delta from previous record: {series.run_dir.name}",
           vmin=DELTA_VMIN, vmax=DELTA_VMAX, cmap=DELTA_CMAP)


def _weight_deltas(series) -> Iterator[tuple[float, np.ndarray]]:
    """前の記録時刻からの重みの変化。最初の時刻には相方が無いので飛ばす。"""
    previous = None
    for window in series.windows:
        values = window.weights()
        if previous is not None:
            yield window.hour, values - previous
        previous = values


def _panel(series, weight_items, out_path: Path, *,
           title: str, vmin: float, vmax: float, cmap: str) -> None:
    """パネル配置の共通部分。**見た目を決めるのは上の 2 つ**で、ここは並べるだけ。"""
    if not weight_items:
        return

    wiring = series.wiring()
    row, col, layout = wiring.row, wiring.col, series.layout
    ordering = style.resolve_ordering(layout, ORDER_AXES)
    n_items = len(weight_items)
    n_cols = min(4, n_items)
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(PANEL_CELL[0] * n_cols, PANEL_CELL[1] * n_rows),
                             squeeze=False)
    last_image = None
    for ax in axes.flat:
        ax.axis("off")

    for ax, (hour, weights) in zip(axes.flat, weight_items):
        ordered = ordering.apply(densify(row, col, weights, layout.total_neurons))
        last_image = ax.imshow(ordered, origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)

        if ordering.enabled:
            style.draw_block_boundaries(ax, ordering, ordered.shape[0])

        ax.set_title(f"{hour:g} h")
        ax.set_xlabel("post")
        ax.set_ylabel("pre")
        ax.axis("on")

    fig.suptitle(title)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    save(fig, out_path, tight_layout=False, dpi=DPI, bbox_inches="tight")
