"""重み行列の imshow 描画。

並べ替えは `order_axes` で指定する (`src.utils.plotting.ordering` を参照)。既定の
`("polarity",)` は興奮性を先頭ブロックに置き、E/I の境界に線を 1 本引く。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting.ordering import (
    DEFAULT_ORDER_AXES,
    Ordering,
    block_ticks,
    resolve_ordering,
)


def _draw_block_boundaries(ax, ordering: Ordering, size: int) -> None:
    """ブロック境界に縦横の線を引く。外側の軸ほど太く描く。"""
    for position, level in ordering.boundaries:
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


def plot_single_weight_matrix(
    weights: np.ndarray,
    layout,
    out_path: Path,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    order_axes: tuple[str, ...] | None = DEFAULT_ORDER_AXES,
) -> None:
    """
    重み行列を可視化する。

    Args:
        weights: 重み行列
        layout: NetworkLayout（並べ替え軸の値をここから取る）
        out_path: 出力ファイルパス
        title: グラフタイトル
        vmin, vmax: カラーバーの範囲
        order_axes: 並べ替えに使う軸を外側から順に。None ならグローバル ID の順のまま。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ordering = resolve_ordering(layout, order_axes)
    ordered = ordering.apply(weights)

    fig, ax = plt.subplots(figsize=(6, 5.4))
    image = ax.imshow(ordered, origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax, cmap="viridis")

    if ordering.enabled:
        _draw_block_boundaries(ax, ordering, ordered.shape[0])
        ticks = block_ticks(ordering, ordered.shape[0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    ax.set_title(title)
    ax.set_xlabel("post neuron id")
    ax.set_ylabel("pre neuron id")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_weight_panel(
    weight_items: list[tuple[float, np.ndarray]],
    layout,
    out_path: Path,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
    order_axes: tuple[str, ...] | None = DEFAULT_ORDER_AXES,
) -> None:
    """
    複数の重み行列をパネル表示する。

    Args:
        weight_items: (時刻, 重み行列) のタプルのリスト
        layout: NetworkLayout（並べ替え軸の値をここから取る）
        out_path: 出力ファイルパス
        title: グラフタイトル
        vmin, vmax: カラーバーの範囲
        cmap: カラーマップ
        order_axes: 並べ替えに使う軸を外側から順に。None ならグローバル ID の順のまま。
    """
    if not weight_items:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ordering = resolve_ordering(layout, order_axes)
    n_items = len(weight_items)
    n_cols = min(4, n_items)
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.9 * n_rows), squeeze=False)
    last_image = None
    for ax in axes.flat:
        ax.axis("off")

    for ax, (hour, weights) in zip(axes.flat, weight_items):
        ordered = ordering.apply(weights)
        last_image = ax.imshow(ordered, origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)

        if ordering.enabled:
            _draw_block_boundaries(ax, ordering, ordered.shape[0])

        ax.set_title(f"{hour:g} h")
        ax.set_xlabel("post")
        ax.set_ylabel("pre")
        ax.axis("on")

    fig.suptitle(title)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
