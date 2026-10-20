"""**結合構造そのもの**の描画 — 誰と誰が、どれくらいの強さで繋がっているか。

3 枚を持つ。いずれも `(view, out_path)` を取る (`src/utils/runview.py` の契約):

- `weight_matrix` / `weight_panel` : 重み行列の imshow。前者はある 1 時点、後者は
  `Series` の記録時刻を並べたもの。
- `connection_mask` : 結合密度を K×K に粗視化した画像。N が大きすぎて
  1 ニューロン 1 ピクセルでは描けない規模のための、上の 2 つの縮約版。
- `empirical_connection_probability` : 同じ結合を **距離の関数** として見た図。
  行列を距離で周辺化したものなので、絵の形は曲線でも主題は結合構造そのもの。
  座標が要るので、`no_space` の run では view が `MissingData` を投げる。

**入力は COO** (row, col と index 整合の値の 1D 配列) — ビルド以降の受け渡しは COO 一本、
という全体の規約に従う。ただし imshow は本質的に (N, N) の画像を要求するので、
**密化はこのモジュールの中だけで起きる**。プロジェクト全体で密な行列を組むのはここだけで、
それは「絵を描く直前のローカルな都合」であって、受け渡しの形式ではない。
N が `DENSE_RENDER_LIMIT` を超えるとメモリに乗らないので、その場合は同じモジュールの
粗視化図 (`plot_connection_mask_coarse`) へ誘導する。

並べ替えは `src.utils.plotting.ordering` が一手に決める。**引数では受けない** ——
ラスターと粗視化図が同じ並びであることが「2 枚を並べて読む」ことの前提なので、
呼び出し側から片方だけ動かせてはいけない。`available_order_axes()` が layout を見て
`("module", "polarity")` か `("polarity",)` を選ぶ。重み行列だけは N×N に module まで
刻むと帯が細くなって読めないので、E/I だけでブロック化する (`WEIGHT_ORDER_AXES`)。

**E/I の示し方は 2 枚で違う。** 重み行列は色を重みの値に使っているので E/I は線で示す。
粗視化図は色そのものが空いているので E/I を **色** (EE/EI/IE/II) に割り当て、線は
モジュール等のブロック境界だけに限る。どちらも「線の種類は 1 つ、色の意味も 1 つ」に
なるようにしている — 太さの違う線が 2 種類入ると格子が読めなくなるため。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
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
    available_order_axes,
    block_ticks,
    draw_block_boundaries,
    resolve_ordering,
)
from src.utils.runview import MissingData

# 密な (N, N) float64 を組んでよい N の上限 (20000^2 x 8B = 3.2 GiB)。
DENSE_RENDER_LIMIT = 20000

# 重み行列の並べ替え軸。**粗視化図やラスターとは別。** 行列は N×N なので、module まで
# 刻むと帯が細くなって読めない。
WEIGHT_ORDER_AXES = DEFAULT_ORDER_AXES

# 見た目。**引数にしない** —— 変えたくなったらここを直す。
SINGLE_FIGSIZE = (6, 5.4)
PANEL_CELL = (4.2, 3.9)       # パネル 1 コマぶんの (幅, 高さ)
VMIN, VMAX = 0.0, 1.0         # 重みは [0, 1] (可塑性の Wmax で正規化済み)
CMAP = "viridis"
DPI = 200
GRID = 256                    # 粗視化の解像度 (K×K)
PROB_N_SRC = 2000             # 距離ビンの分母に使う送信ニューロンのサンプル数
PROB_NUM_BINS = 40
PROB_SEED = 0


def densify(row: np.ndarray, col: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """COO を imshow 用の (size, size) 行列へ起こす。**描画直前のローカルな密化。**

    結合が無い箇所は 0 で埋まるので、**統計には使わないこと**
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
    """ある 1 時点の重み行列。build 直後の初期重みでも記録窓の重みでも同じ呼び出し。"""
    coo = view.coo()
    layout = view.layout
    ordering = resolve_ordering(layout, WEIGHT_ORDER_AXES)
    ordered = ordering.apply(densify(coo.row, coo.col, coo.weights, layout.total_neurons))

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    image = ax.imshow(ordered, origin="upper", interpolation="nearest",
                      vmin=VMIN, vmax=VMAX, cmap=CMAP)

    if ordering.enabled:
        draw_block_boundaries(ax, ordering, ordered.shape[0])
        ticks = block_ticks(ordering, ordered.shape[0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    ax.set_title(f"Weight matrix {view.hour:g} h" if hasattr(view, "hour")
                 else "Weight matrix")
    ax.set_xlabel("post neuron id")
    ax.set_ylabel("pre neuron id")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="weight")
    save_figure(fig, out_path, dpi=DPI)


def weight_panel(series, out_path: Path) -> None:
    """記録時刻ごとの重み行列を時系列に並べたパネル。"""
    items = [(window.hour, window.weights()) for window in series.windows]
    if not items:
        raise MissingData("windows", "記録窓が 1 つもありません")

    wiring = series.wiring()
    layout = series.layout
    ordering = resolve_ordering(layout, WEIGHT_ORDER_AXES)
    n_cols = min(4, len(items))
    n_rows = int(np.ceil(len(items) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(PANEL_CELL[0] * n_cols, PANEL_CELL[1] * n_rows),
                             squeeze=False)
    last_image = None
    for ax in axes.flat:
        ax.axis("off")

    for ax, (hour, weights) in zip(axes.flat, items):
        ordered = ordering.apply(densify(wiring.row, wiring.col, weights,
                                         layout.total_neurons))
        last_image = ax.imshow(ordered, origin="upper", interpolation="nearest",
                               vmin=VMIN, vmax=VMAX, cmap=CMAP)
        if ordering.enabled:
            draw_block_boundaries(ax, ordering, ordered.shape[0])
        ax.set_title(f"{hour:g} h")
        ax.set_xlabel("post")
        ax.set_ylabel("pre")
        ax.axis("on")

    fig.suptitle(f"Weight matrix timeline: {series.run_dir.name}")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    save_figure(fig, out_path, tight_layout=False, dpi=DPI, bbox_inches="tight")


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


def connection_mask(view, out_path: Path) -> None:
    """結合マスクを K×K に粗視化した密度画像。

    (40000, 40000) の imshow は不可能かつ視覚的にも無意味なので、並べ替えた表示順位の
    軸上で K×K のセルに落とし、セルごとの結合密度 (実結合数 / セル内の全ペア数) を描く。

    **並べ替え軸は layout が決める** (`available_order_axes()`)。module 軸を持つ run なら
    「モジュールで切って各モジュール内で E→I」、持たなければ E/I ブロックだけ。
    ラスターも同じ選び方をするので、2 枚の y 軸は必ず揃う。最外ブロックには軸の値
    (`M0`, `M1`, …) が目盛りとして入る。

    **色 = E/I ブロック (EE/EI/IE/II)、濃さ = 結合確率、線 = 最外ブロックの境界。**
    E/I の切り替わりには線を引かない (色が示しているため)。カラーバーの代わりに
    E/I ブロックの凡例を出し、濃さのスケールをその見出しに書く。

    セル数 K は `GRID` と N の小さい方。
    """
    wiring = view.wiring()
    row, col = wiring.row, wiring.col
    layout = view.layout
    total_neurons = view.total_neurons

    ordering = resolve_ordering(layout, available_order_axes(layout))
    grid = int(min(GRID, total_neurons))
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

    # 色 = E/I ブロック、濃さ = 結合確率。カラーバー 1 本では E/I を表せないので、
    # スカラーの cmap ではなく RGB 画像を自分で組んで凡例で説明する。
    rgb, max_density = _block_colored_density(density, cell_of_rank, per_cell,
                                              excitatory_flags(layout, total_neurons))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(rgb, origin="upper", interpolation="nearest")
    # 線は**モジュール等のブロック境界だけ**。E/I の切り替わりは色が示しているので、
    # 線の種類を増やさない (2 種類の線が混ざると格子が読めなくなる)。
    draw_block_boundaries(ax, ordering, grid, scale=scale, skip=("polarity",))

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
    ax.set_title("Connection mask (coarse-grained)\n"
                 f"{np.asarray(row).size} synapses, {total_neurons} neurons")
    # カラーバーの代わりに凡例。連続量は「濃さ」1 次元しかないので、凡例のタイトルに
    # そのスケール (白 = 0、最も濃い色 = max) を書いておけば読み取れる。
    ax.legend(
        handles=[Patch(facecolor=BLOCK_COLORS[name], edgecolor="none", label=name)
                 for name in BLOCK_ORDER],
        title=f"pre→post block\nsaturation: 0 – {max_density:.3f}",
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, title_fontsize=8,
        frameon=False,
    )
    save_figure(fig, out_path, dpi=DPI)


def _block_colored_density(density: np.ndarray, cell_of_rank: np.ndarray,
                           per_cell: np.ndarray, is_exc: np.ndarray) -> tuple[np.ndarray, float]:
    """粗視化密度を「色 = E/I ブロック、濃さ = 密度」の RGB 画像にする。

    セルの E/I は多数決で決める (表示順に並んでいればセルはほぼ純粋に片側になる)。
    色は白 (密度 0) からブロック色 (密度が最大) への線形補間。

    Returns:
        (RGB 画像 (K, K, 3), 濃さの上限として使った密度)
    """
    exc_per_cell = np.bincount(cell_of_rank[is_exc], minlength=per_cell.size).astype(np.float64)
    cell_is_exc = exc_per_cell >= (per_cell - exc_per_cell)   # 同数なら興奮性側へ

    src, tgt = cell_is_exc[:, None], cell_is_exc[None, :]
    # BLOCK_ORDER = ("EE", "EI", "IE", "II") の添字へ落とす。
    block_index = np.where(src, np.where(tgt, 0, 1), np.where(tgt, 2, 3))
    palette = np.array([to_rgb(BLOCK_COLORS[name]) for name in BLOCK_ORDER])

    max_density = float(density.max()) if density.size else 0.0
    alpha = (density / max_density) if max_density > 0 else np.zeros_like(density)
    return 1.0 - alpha[..., None] * (1.0 - palette[block_index]), max_density


def empirical_connection_probability(view, out_path: Path) -> None:
    """距離ビンごとの実測結合確率を E/I ブロック別に描く。距離依存の結合則の検算。

    送信側を `PROB_N_SRC` 個サンプルし、そのサンプルの中で分母 (使えるペア数) と
    分子 (実結合) を数えるので、比は不偏な結合確率の推定になる。

    config の connection プロファイルが sigma_xy / p0_xy を持てば、理論曲線
    p0*exp(-d^2/2σ^2) を重ねる。
    """
    coords = np.asarray(view.coords(), dtype=np.float64)
    wiring = view.wiring()
    row, col = wiring.row, wiring.col
    layout = view.layout
    connection_config = view.config.network.connection

    total = coords.shape[0]
    rng = np.random.default_rng(PROB_SEED)
    is_exc = excitatory_flags(layout, total)

    sources = np.sort(rng.choice(total, size=min(PROB_N_SRC, total), replace=False))
    selected = np.zeros(total, dtype=bool)
    selected[sources] = True

    distances = cdist(coords[sources, :2], coords[:, :2])
    max_distance = float(distances.max()) if distances.size else 1.0
    edges = np.linspace(0.0, max_distance, PROB_NUM_BINS + 1)
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
    ax.set_title("Empirical connection probability\n"
                 f"{sources.size} source neurons sampled")
    ax.legend(fontsize=7)
    save_figure(fig, out_path, dpi=DPI)
