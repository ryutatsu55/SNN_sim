"""結合構造そのものの図。

- `connection_mask` : 群ごとに粗視化した結合密度。どの群からどの群へ、が見える
- `empirical_connection_probability` : 距離に対する結合確率。距離依存の結合則の検算

どちらも「何本つながっているか」を見る図で、重みの値は見ない (それは `weight_matrix.py`
と `synapse_hist.py`)。
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
from scripts.tools.runview import MissingData
from scripts.develop.figures import save, style
from scripts.develop.figures.style import BLOCK_COLORS, Ordering


# 密な (N, N) float64 を組んでよい N の上限 (20000^2 x 8B = 3.2 GiB)。
DENSE_RENDER_LIMIT = 20000
# 最外ブロックの名前を目盛りに出す上限。これを超えると文字が潰れるので番号のままにする。
MAX_BLOCK_TICKS = 16


# 保存時の解像度。**引数にしない** —— 変えたくなったらここを直す。
DPI = 200
GRID = 256                 # 粗視化の解像度 (K×K)
PROB_N_SRC = 2000          # 距離ビンの分母に使う送信ニューロンのサンプル数
PROB_NUM_BINS = 40
PROB_SEED = 0
MASK_TITLE = "Connection mask (coarse-grained)"
PROB_TITLE = "Empirical connection probability"


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

def connection_mask(built, out_path: Path) -> None:
    """結合マスクを K×K に粗視化した密度画像。

    (40000, 40000) の imshow は不可能かつ視覚的にも無意味なので、並べ替えた表示順位の
    軸上で K×K のセルに落とし、セルごとの結合密度 (実結合数 / セル内の全ペア数) を描く。

    **並べ替え軸はラスターと同じ** (`style.available_order_axes()`)。`("polarity",)`
    (E/I ブロック) でも `("module",)` (モジュールごとのブロック) でも
    `("module", "polarity")` (モジュールで切って各モジュール内で E→I) でも同じ形で効く。
    最外ブロックには軸の値 (`M0`, `M1`, …) が目盛りとして入る。

    **色 = E/I ブロック (EE/EI/IE/II)、濃さ = 結合確率、線 = 最外ブロックの境界**。
    E/I の切り替わりには線を引かない (色が既にそれを示しているので、線が 2 種類あると
    格子が読めなくなる)。カラーバーは E/I と密度の 2 つを同時に表せないので出さず、
    代わりに E/I ブロックの凡例を出して濃さのスケールをその見出しに書く。

    """
    wiring = built.wiring()
    row, col = wiring.row, wiring.col
    layout = built.layout
    total_neurons = built.total_neurons

    ordering = style.resolve_ordering(layout, style.available_order_axes(layout))
    cells = int(min(GRID, total_neurons))
    if cells < 1:
        raise ValueError("粗視化の解像度は 1 以上である必要があります。")

    rank = (ordering.rank if ordering.enabled
            else np.arange(total_neurons, dtype=np.int64))

    # 表示順位 -> セル番号
    scale = cells / total_neurons
    cell_of_rank = (rank.astype(np.float64) * scale).astype(np.int64)
    cell_of_rank = np.clip(cell_of_rank, 0, cells - 1)

    counts = np.zeros((cells, cells), dtype=np.float64)
    np.add.at(counts, (cell_of_rank[np.asarray(row, dtype=np.int64)],
                       cell_of_rank[np.asarray(col, dtype=np.int64)]), 1.0)

    # セルごとの「ありうるペア数」で割って密度にする (セルの大きさが均等でない場合に効く)
    per_cell = np.bincount(cell_of_rank, minlength=cells).astype(np.float64)
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
    style.draw_block_boundaries(ax, ordering, cells, scale=scale, skip=("polarity",))

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
    ax.set_xlabel(f"Target (grouped by {grouped}, {cells} cells)")
    ax.set_ylabel(f"Source (grouped by {grouped}, {cells} cells)")
    ax.set_title(f"{MASK_TITLE}\n{np.asarray(row).size} synapses, {total_neurons} neurons")
    # カラーバーの代わりに凡例。連続量は「濃さ」1 次元しかないので、凡例のタイトルに
    # そのスケール (白 = 0、最も濃い色 = max) を書いておけば読み取れる。
    ax.legend(
        handles=[Patch(facecolor=BLOCK_COLORS[name], edgecolor="none", label=name)
                 for name in BLOCK_ORDER],
        title=f"pre→post block\nsaturation: 0 – {max_density:.3f}",
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, title_fontsize=8,
        frameon=False,
    )
    save(fig, out_path, dpi=DPI)

def _block_colored_density(density: np.ndarray, cell_of_rank: np.ndarray,
                           per_cell: np.ndarray, is_exc: np.ndarray) -> tuple[np.ndarray, float]:
    """粗視化密度を「色 = E/I ブロック、濃さ = 密度」の RGB 画像にする。

    セルは表示順の連続した塊なので、`("module", "polarity")` で並べていれば
    ほぼ純粋に E か I のどちらかになる。境目をまたぐセルだけは多数決で決める。
    白 (密度 0) から そのセルのブロック色 (密度が最大) への線形補間なので、
    「どのブロックか」と「どれくらい繋がっているか」が 1 枚で両立する。

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

def empirical_connection_probability(built, out_path: Path) -> None:
    """距離ビンごとの実測結合確率を E/I ブロック別に描き、理論曲線を重ねる。

    全ペアの距離分布 (分母) は N^2 なので、送信側を `PROB_N_SRC` 個サンプルして
    そのサンプルに対してのみ `cdist` で分母のヒストグラムを作る。分子は同じサンプルの
    実結合のみを数えるので、比は不偏な結合確率の推定になる。

    `config.network.connection` に sigma_xy / p0_xy があれば理論曲線
    p0*exp(-d^2/2σ^2) を重ねる。
    """
    wiring = built.wiring()
    if wiring.num_synapses == 0:
        raise MissingData("synapses", "結合が 1 本もありません")
    row, col = wiring.row, wiring.col
    layout = built.layout
    connection_config = built.config.network.connection

    coords = np.asarray(built.coords(), dtype=np.float64)
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
    ax.set_title(f"{PROB_TITLE}\n{sources.size} source neurons sampled")
    ax.legend(fontsize=7)
    save(fig, out_path, dpi=DPI)
