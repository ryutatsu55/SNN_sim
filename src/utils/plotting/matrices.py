"""**結合構造そのもの**の描画 — 誰と誰が、どれくらいの強さで繋がっているか。

3 枚を持つ。いずれも座標を使わない (空間に置いた図は `network.py`):

- `plot_single_weight_matrix` / `plot_weight_panel` : 重み行列の imshow。
- `plot_connection_mask_coarse` : 結合密度を K×K に粗視化した画像。N が大きすぎて
  1 ニューロン 1 ピクセルでは描けない規模のための、上の 2 つの縮約版。
- `plot_empirical_connection_probability` : 同じ結合を **距離の関数** として見た図。
  行列を距離で周辺化したものなので、絵の形は曲線でも主題は結合構造そのもの。

**入力は COO** (row, col と index 整合の値の 1D 配列) — ビルド以降の受け渡しは COO 一本、
という全体の規約に従う。ただし imshow は本質的に (N, N) の画像を要求するので、
**密化はこのモジュールの中だけで起きる**。プロジェクト全体で密な行列を組むのはここだけで、
それは「絵を描く直前のローカルな都合」であって、受け渡しの形式ではない。
N が `DENSE_RENDER_LIMIT` を超えるとメモリに乗らないので、その場合は同じモジュールの
粗視化図 (`plot_connection_mask_coarse`) へ誘導する。

並べ替えは **3 枚とも同じ `order_axes`** で指定する (`src.utils.plotting.ordering` を参照)。
既定の `("polarity",)` は興奮性を先頭ブロックに置き、E/I の境界に線を 1 本引く。
`("module",)` ならモジュールごとのブロックに、`("module", "polarity")` ならモジュールで
切ったうえで各モジュール内が E→I になる。粗視化図もこの仕組みに乗っているので、
E/I 専用だった頃の `display_rank()` は無くなった。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from src.utils.analysis.weights import (
    BLOCK_ORDER,
    block_masks,
    excitatory_flags,
    synapse_distances,
)
from src.utils.plotting.common import BLOCK_COLORS, save_figure
from src.utils.plotting.ordering import (
    DEFAULT_ORDER_AXES,
    Ordering,
    block_ticks,
    resolve_ordering,
)

# 密な (N, N) float64 を組んでよい N の上限 (20000^2 x 8B = 3.2 GiB)。
DENSE_RENDER_LIMIT = 20000


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
            " 粗視化図 (plot_connection_mask_coarse) を使ってください。"
        )
    values = np.asarray(values).reshape(-1)
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    if not (row.size == col.size == values.size):
        raise ValueError("row / col / values の長さが一致しません。")
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[row, col] = values
    return matrix


def _draw_block_boundaries(ax, ordering: Ordering, size: int, *, scale: float = 1.0) -> None:
    """ブロック境界に縦横の線を引く。外側の軸ほど太く描く。

    `scale` は「表示位置 (ニューロン単位) → 画像の画素」の倍率。1 ニューロン 1 画素の
    重み行列では 1.0、粗視化図では `grid / total_neurons` になる。
    """
    for position, level in ordering.boundaries:
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


def plot_single_weight_matrix(
    row: np.ndarray,
    col: np.ndarray,
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
        row, col: 各結合の送信/受信グローバルID (1D)
        weights: 各結合の重み (1D, row/col と index 整合)
        layout: NetworkLayout（並べ替え軸の値と行列サイズをここから取る）
        out_path: 出力ファイルパス
        title: グラフタイトル
        vmin, vmax: カラーバーの範囲
        order_axes: 並べ替えに使う軸を外側から順に。None ならグローバル ID の順のまま。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ordering = resolve_ordering(layout, order_axes)
    ordered = ordering.apply(densify(row, col, weights, layout.total_neurons))

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
    row: np.ndarray,
    col: np.ndarray,
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

    結合構造は記録の間で変わらないので row/col は 1 組だけ受け取り、時刻ごとに変わる
    値ベクトルを `weight_items` で渡す。

    Args:
        row, col: 各結合の送信/受信グローバルID (1D, 全時刻で共通)
        weight_items: (時刻, 重みの値ベクトル) のタプルのリスト
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
        ordered = ordering.apply(densify(row, col, weights, layout.total_neurons))
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


# 最外ブロックの名前を目盛りに出す上限。これを超えると文字が潰れるので番号のままにする。
MAX_BLOCK_TICKS = 16


def _outer_block_labels(layout, ordering: Ordering, total: int) -> list[tuple[float, str]]:
    """最外ブロック (order_axes の先頭の軸) の中心位置とラベル名。

    位置は**表示位置** (ニューロン単位) なので、粗視化図では呼び出し側でセルへ換算する。
    """
    if layout is None or not ordering.enabled or not ordering.axes:
        return []
    values = layout.labels(ordering.axes[0])[ordering.order]
    edges = [0, *ordering.positions(level=0), total]
    return [(0.5 * (edges[i] + edges[i + 1]), str(values[edges[i]]))
            for i in range(len(edges) - 1)]


def plot_connection_mask_coarse(
    row: np.ndarray,
    col: np.ndarray,
    layout,
    total_neurons: int,
    out_path: Path,
    title: str = "Connection mask (coarse-grained)",
    grid: int = 256,
    order_axes: tuple[str, ...] | None = DEFAULT_ORDER_AXES,
) -> None:
    """結合マスクを K×K に粗視化した密度画像。

    (40000, 40000) の imshow は不可能かつ視覚的にも無意味なので、並べ替えた表示順位の
    軸上で K×K のセルに落とし、セルごとの結合密度 (実結合数 / セル内の全ペア数) を描く。

    **並べ替え軸は `order_axes` で選ぶ** — 他の行列図やラスターと同じ
    `src.utils.plotting.ordering` の仕組みに乗っているので、`("polarity",)` (既定、E/I
    ブロック) でも `("module",)` (モジュールごとのブロック) でも
    `("module", "polarity")` (モジュールで切って各モジュール内で E→I) でも同じ形で効く。
    ブロック境界の線は入れ子の階層に応じて太さが変わり、最外ブロックには軸の値
    (`M0`, `M1`, … / `excitatory`, `inhibitory`) が目盛りとして入る。

    Args:
        row, col: 各結合の送信/受信グローバルID (1D)。
        layout: NetworkLayout。並べ替えと軸ラベルに使う。
        total_neurons: N。セルの大きさの換算に使う。
        grid: 1 辺のセル数 K。N より大きくしても意味がないので N で頭打ちにする。
        order_axes: 並べ替えに使う軸を外側から順に。None / 空ならグローバル ID の順。
    """
    ordering = resolve_ordering(layout, order_axes)
    grid = int(min(grid, total_neurons))
    if grid < 1:
        raise ValueError("grid must be >= 1")

    rank = (ordering.rank if ordering.enabled
            else np.arange(total_neurons, dtype=np.int64))

    # 表示順位 -> セル番号
    scale = grid / total_neurons
    cell_of_rank = (rank.astype(np.float64) * scale).astype(np.int64)
    cell_of_rank = np.clip(cell_of_rank, 0, grid - 1)

    counts = np.zeros((grid, grid), dtype=np.float64)
    np.add.at(counts, (cell_of_rank[np.asarray(row, dtype=np.int64)],
                       cell_of_rank[np.asarray(col, dtype=np.int64)]), 1.0)

    # セルごとの「ありうるペア数」で割って密度にする (セルの大きさが均等でない場合に効く)
    per_cell = np.bincount(cell_of_rank, minlength=grid).astype(np.float64)
    possible = np.outer(per_cell, per_cell)
    density = np.divide(counts, possible, out=np.zeros_like(counts), where=possible > 0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(density, origin="upper", interpolation="nearest", cmap="viridis")
    _draw_block_boundaries(ax, ordering, grid, scale=scale)

    # 最外ブロックの名前を目盛りに。多すぎると潰れるのでセル番号のままにする。
    blocks = _outer_block_labels(layout, ordering, total_neurons)
    if blocks and len(blocks) <= MAX_BLOCK_TICKS:
        ticks = [center * scale - 0.5 for center, _ in blocks]
        names = [name for _, name in blocks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(ticks)
        ax.set_yticklabels(names)

    grouped = (" > ".join(ordering.axes) if ordering.enabled else "global ID order")
    ax.set_xlabel(f"Target (grouped by {grouped}, {grid} cells)")
    ax.set_ylabel(f"Source (grouped by {grouped}, {grid} cells)")
    ax.set_title(f"{title}\n{np.asarray(row).size} synapses, {total_neurons} neurons")
    fig.colorbar(image, ax=ax, label="connection probability")
    save_figure(fig, out_path)


def plot_empirical_connection_probability(
    coords: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    layout,
    out_path: Path,
    connection_config=None,
    title: str = "Empirical connection probability",
    n_src: int = 2000,
    num_bins: int = 40,
    seed: int = 0,
) -> None:
    """距離ビンごとの実測結合確率を E/I ブロック別に描き、理論曲線を重ねる。

    全ペアの距離分布 (分母) は N^2 なので、送信側を `n_src` 個サンプルして
    そのサンプルに対してのみ `cdist` で分母のヒストグラムを作る。分子は同じサンプルの
    実結合のみを数えるので、比は不偏な結合確率の推定になる。

    `connection_config` に sigma_xy / p0_xy があれば理論曲線 p0*exp(-d^2/2σ^2) を重ねる。
    """
    coords = np.asarray(coords, dtype=np.float64)
    total = coords.shape[0]
    rng = np.random.default_rng(seed)
    is_exc = excitatory_flags(layout, total)

    sources = np.sort(rng.choice(total, size=min(n_src, total), replace=False))
    selected = np.zeros(total, dtype=bool)
    selected[sources] = True

    distances = cdist(coords[sources, :2], coords[:, :2])
    max_distance = float(distances.max()) if distances.size else 1.0
    edges = np.linspace(0.0, max_distance, num_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    src_exc_grid = is_exc[sources][:, None]
    tgt_exc_grid = is_exc[None, :]
    denominator_masks = {
        "EE": src_exc_grid & tgt_exc_grid,
        "EI": src_exc_grid & ~tgt_exc_grid,
        "IE": ~src_exc_grid & tgt_exc_grid,
        "II": ~src_exc_grid & ~tgt_exc_grid,
    }

    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    keep = selected[row]
    connected_distance = synapse_distances(coords, row[keep], col[keep])
    numerator_masks = block_masks(row[keep], col[keep], is_exc)

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in BLOCK_ORDER:
        denominator = np.histogram(distances[denominator_masks[name]], bins=edges)[0]
        numerator = np.histogram(connected_distance[numerator_masks[name]], bins=edges)[0]
        valid = denominator > 0
        if not valid.any():
            continue
        probability = np.zeros_like(centres)
        probability[valid] = numerator[valid] / denominator[valid]
        ax.plot(centres[valid], probability[valid], color=BLOCK_COLORS[name],
                lw=1.4, marker="o", ms=2.5, label=f"{name} (measured)")

        if connection_config is not None:
            sigma = getattr(connection_config, f"sigma_{name.lower()}", None)
            p0 = getattr(connection_config, f"p0_{name.lower()}", None)
            if sigma is not None and p0 is not None:
                theory = p0 * np.exp(-(centres ** 2) / (2.0 * float(sigma) ** 2))
                ax.plot(centres, theory, color=BLOCK_COLORS[name], lw=1.0, ls="--", alpha=0.7)

    ax.plot([], [], color="gray", ls="--", lw=1.0, label="theory p0·exp(-d²/2σ²)")
    ax.set_xlabel("Distance [um]")
    ax.set_ylabel("Connection probability")
    ax.set_title(f"{title}\n{sources.size} source neurons sampled")
    ax.legend(fontsize=7)
    save_figure(fig, out_path)
