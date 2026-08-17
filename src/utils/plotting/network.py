"""ネットワーク構造の可視化。

2 つの規模帯をカバーする:

- `network` : ニューロンの空間配置を矢印付きのグラフとして描く。ニューロンとエッジを
  サンプリングするので大規模でも破綻しないが、密な (N, N) 重み行列を必要とする。
- 残りの関数: COO (row, col) と 1D の重み/遅延配列だけを受け取り、サンプリングと粗視化で
  「見て意味のある」図に落とす。N が数万・シナプスが数千万でも通る。

いずれも E/I の分類は `layout.ids_by("polarity")` から得る。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from src.utils.analysis.weights import BLOCK_ORDER, block_masks, excitatory_flags

# 送信種別 × 受信種別の描画色 (ブロック名の正準順は analysis.weights.BLOCK_ORDER)
BLOCK_COLORS = {
    "EE": "tab:red",
    "EI": "tab:orange",
    "IE": "tab:blue",
    "II": "tab:purple",
}


def display_rank(layout, total_neurons: int) -> tuple[np.ndarray, int]:
    """グローバルID -> 表示順位 (興奮性が先頭ブロック) の写像と、興奮性の数を返す。"""
    rank = layout.rank_by("polarity")
    return rank, int(layout.ids_by("polarity")["excitatory"].size)


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def network(weights: np.ndarray, coords: np.ndarray, config, layout=None, node_size=10,
            title="network", save_path=".", n_sample=500, max_edges=4000, seed=0):
    """
    ニューロンの空間配置と重み行列からネットワーク構造を可視化する。

    空間ネットワーク図を担う唯一の関数。大規模ネットワークでも破綻しないよう、
    ニューロンを n_sample 個サンプリングし、両端がサンプルに含まれる結合だけを
    描画する (さらに max_edges 本へ間引く)。エッジは矢印付きで、重みの符号で色分け
    (正=興奮性=赤 / 負=抑制性=青)、絶対値に応じて線の太さを変える。

    Parameters:
        weights (np.ndarray): 結合重み行列。形状は (N, N)。
        coords (np.ndarray): ニューロンの座標配列。形状は (N, 3)。
        config: AppConfig。矩形空間 (x_range/y_range) なら軸範囲に使う。
        layout: NetworkLayout。渡すとノードを E/I で色分けする (省略時は一色)。
        node_size (int): ニューロン(ノード)の描画サイズ。
        title (str): グラフ/ファイル名。
        save_path (str): 画像の保存先ディレクトリ。
        n_sample (int): 描画に用いるニューロンのサンプリング数。
        max_edges (int): 描画するエッジ数の上限 (annotate ループを抑える)。
        seed (int): サンプリングの乱数シード。
    """
    weights = np.asarray(weights)
    coords = np.asarray(coords)
    N = weights.shape[0]
    rng = np.random.default_rng(seed)

    # Z軸が存在する場合でも、今回は2D平面(X, Y)への投影として扱う
    x = coords[:, 0]
    y = coords[:, 1]

    # --- ニューロンのサンプリング ---
    sample = np.sort(rng.choice(N, size=min(n_sample, N), replace=False))
    in_sample = np.zeros(N, dtype=bool)
    in_sample[sample] = True

    # --- E/I 判定 (layout があれば興奮性を True に) ---
    is_exc = None
    if layout is not None:
        exc_ids = np.asarray(layout.ids_by("polarity").get("excitatory", []), dtype=np.int64)
        is_exc = np.zeros(N, dtype=bool)
        is_exc[exc_ids] = True

    fig, ax = plt.subplots(figsize=(12, 10))

    # --- ノード描画 (layout があれば E/I で色分け) ---
    if is_exc is not None:
        exc_sample = sample[is_exc[sample]]
        inh_sample = sample[~is_exc[sample]]
        ax.scatter(x[exc_sample], y[exc_sample], s=node_size, color='tab:red',
                   edgecolors='black', zorder=3, label='excitatory')
        ax.scatter(x[inh_sample], y[inh_sample], s=node_size, color='tab:blue',
                   edgecolors='black', zorder=3, label='inhibitory')
        ax.legend(fontsize=9, markerscale=1.5)
    else:
        ax.scatter(x[sample], y[sample], s=node_size, color='darkgray',
                   edgecolors='black', zorder=3)

    # 描画用のスケール計算（太さの正規化用）
    abs_max = np.max(np.abs(weights)) if weights.size else 0.0
    max_weight = abs_max if abs_max > 0 else 1.0
    # 結合（エッジ）を抽出し、両端がサンプルに含まれるものだけ残す
    sources, targets = np.where(np.abs(weights) != 0)
    keep = in_sample[sources] & in_sample[targets]
    sources, targets = sources[keep], targets[keep]
    # エッジが多すぎる場合はさらに max_edges 本へ間引く (annotate は1本ずつ描くため)
    if sources.size > max_edges:
        pick = rng.choice(sources.size, size=max_edges, replace=False)
        sources, targets = sources[pick], targets[pick]

    # 矢印がノードの中心に刺さるのを防ぐためのマージン計算
    # (scatterの s は面積なので、半径は平方根に比例)
    node_margin = np.sqrt(node_size) * 0.8

    for s, t in zip(sources, targets):

        # 重みの強さに応じて線の太さを変更 (最大2.0)
        w = weights[s, t]
        lw = (abs(w) / max_weight) * 2.0

        # エッジ色は出力元ノード (source) の色に揃える。
        # layout があれば興奮性=赤 / 抑制性=青、無ければノードと同じ灰色。
        if is_exc is None:
            color = (0.4, 0.4, 0.4, 0.5)  # darkgray 相当 (alpha=0.5)
        elif is_exc[s]:
            color = (0.8, 0.2, 0.2, 0.5)  # Red (excitatory source)
        else:
            color = (0.2, 0.2, 0.8, 0.5)  # Blue (inhibitory source)

        # ax.annotate を用いて矢印を描画
        ax.annotate(
            "",
            xy=(x[t], y[t]),       # 終点 (Target)
            xytext=(x[s], y[s]),   # 始点 (Source)
            arrowprops=dict(
                arrowstyle="->, head_length=0.4, head_width=0.2", # 矢印の形状
                color=color,
                linewidth=lw,
                shrinkA=node_margin,  # 始点側の隙間（ノードと重ならないように）
                shrinkB=node_margin,  # 終点側の隙間（矢印の先がノードに隠れないように）
                # 双方向の結合が重ならないよう、線を少しカーブさせる (rad=0.1)
                connectionstyle="arc3,rad=0.1"
            ),
            zorder=1
        )

    ax.set_aspect('equal')
    ax.set_title(f"{title}\n{sample.size} neurons sampled, {sources.size} edges drawn")
    ax.set_xlabel("X Coordinate [um]")
    ax.set_ylabel("Y Coordinate [um]")
    # x_range/y_range を持つ矩形空間なら明示的に範囲を合わせる。
    # 持たない空間 (random_circle_2d など) は座標からの自動スケールに任せる。
    space_cfg = config.network.space
    x_range = getattr(space_cfg, "x_range", None)
    y_range = getattr(space_cfg, "y_range", None)
    if x_range is not None and y_range is not None:
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
    else:
        ax.margins(0.05)

    plt.tight_layout()

    plt.savefig(f"{save_path}/{title}.png", dpi=300, bbox_inches='tight')
    print(f"Network visualization saved to {save_path}/{title}.png")

    plt.close()


def plot_delay_distribution(
    delays_ms: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    layout,
    total_neurons: int,
    out_path: Path,
    title: str = "Delay distribution",
    bins: int = 80,
) -> None:
    """実在する結合上の伝播遅延のヒストグラム (全体 + E/I ブロック別)。

    結合が無い箇所は行列上 0 で埋まるため、必ず COO (= 実結合のみ) を渡すこと。
    """
    delays = np.asarray(delays_ms, dtype=np.float64)
    is_exc = excitatory_flags(layout, total_neurons)
    masks = block_masks(row, col, is_exc)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(delays, bins=bins, color="black")
    axes[0].set_xlabel("Delay [ms]")
    axes[0].set_ylabel("Number of synapses")
    axes[0].set_title(f"All synapses (n={delays.size})\n"
                      f"mean={delays.mean():.2f} ms, max={delays.max():.2f} ms"
                      if delays.size else "All synapses (empty)")

    edges = np.histogram_bin_edges(delays, bins=bins) if delays.size else np.linspace(0, 1, bins)
    for name in BLOCK_ORDER:
        block = delays[masks[name]]
        if block.size:
            axes[1].hist(block, bins=edges, histtype="step", lw=1.4,
                         color=BLOCK_COLORS[name], label=f"{name} (n={block.size})")
    axes[1].set_xlabel("Delay [ms]")
    axes[1].set_ylabel("Number of synapses")
    axes[1].set_title("By connection type")
    axes[1].legend(fontsize=7)

    fig.suptitle(title)
    _save(fig, out_path)


def plot_connection_mask_coarse(
    row: np.ndarray,
    col: np.ndarray,
    layout,
    total_neurons: int,
    out_path: Path,
    title: str = "Connection mask (coarse-grained)",
    grid: int = 256,
) -> None:
    """結合マスクを K×K に粗視化した密度画像。

    (40000, 40000) の imshow は不可能かつ視覚的にも無意味なので、興奮性を先頭に
    並べ替えた表示順位の軸上で K×K のセルに落とし、セルごとの結合密度
    (実結合数 / セル内の全ペア数) を描く。E/I ブロック構造はそのまま残る。
    """
    rank, n_exc = display_rank(layout, total_neurons)
    grid = int(min(grid, total_neurons))
    if grid < 1:
        raise ValueError("grid must be >= 1")

    # 表示順位 -> セル番号
    cell_of_rank = (rank.astype(np.float64) * grid / total_neurons).astype(np.int64)
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
    boundary = n_exc * grid / total_neurons
    if 0 < boundary < grid:
        ax.axhline(boundary - 0.5, color="white", lw=0.8, ls="--")
        ax.axvline(boundary - 0.5, color="white", lw=0.8, ls="--")
    ax.set_xlabel(f"Target (excitatory first, {grid} cells)")
    ax.set_ylabel(f"Source (excitatory first, {grid} cells)")
    ax.set_title(f"{title}\n{np.asarray(row).size} synapses, {total_neurons} neurons")
    fig.colorbar(image, ax=ax, label="connection probability")
    _save(fig, out_path)


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
    connected_distance = np.linalg.norm(coords[row[keep], :2] - coords[col[keep], :2], axis=1)
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
    _save(fig, out_path)


def plot_weight_distributions(
    hours: list[float],
    weight_arrays: list[np.ndarray],
    out_path: Path,
    row: np.ndarray | None = None,
    col: np.ndarray | None = None,
    layout=None,
    total_neurons: int | None = None,
    title: str = "Weight distribution over time",
    bins: int = 80,
) -> None:
    """各計測時刻の重み分布を 1 枚に重ね描きし、E/I ブロック別のパネルも添える。

    row/col/layout/total_neurons を渡すとブロック別パネルを描く。省略した場合は
    全体のヒストグラムのみ。
    """
    if len(hours) != len(weight_arrays):
        raise ValueError("hours と weight_arrays の長さが一致しません。")

    has_blocks = row is not None and col is not None and layout is not None and total_neurons
    masks = block_masks(row, col, excitatory_flags(layout, total_neurons)) if has_blocks else None

    num_panels = 1 + (len(BLOCK_ORDER) if has_blocks else 0)
    columns = min(num_panels, 3)
    rows_needed = int(np.ceil(num_panels / columns))
    fig, axes = plt.subplots(rows_needed, columns,
                             figsize=(4.2 * columns, 3.4 * rows_needed), squeeze=False)
    flat_axes = axes.ravel()

    all_values = np.concatenate([np.asarray(w, dtype=np.float64) for w in weight_arrays]) \
        if weight_arrays else np.array([0.0, 1.0])
    edges = np.histogram_bin_edges(all_values, bins=bins)
    colours = plt.cm.viridis(np.linspace(0, 0.9, max(len(hours), 1)))

    for hour, weights, colour in zip(hours, weight_arrays, colours):
        values = np.asarray(weights, dtype=np.float64)
        flat_axes[0].hist(values, bins=edges, histtype="step", lw=1.4,
                          color=colour, label=f"{hour:g} h")
    flat_axes[0].set_title("All synapses")
    flat_axes[0].set_xlabel("Weight")
    flat_axes[0].set_ylabel("Number of synapses")
    flat_axes[0].legend(fontsize=7)

    if has_blocks:
        for panel, name in enumerate(BLOCK_ORDER, start=1):
            axis = flat_axes[panel]
            for hour, weights, colour in zip(hours, weight_arrays, colours):
                values = np.asarray(weights, dtype=np.float64)[masks[name]]
                if values.size:
                    axis.hist(values, bins=edges, histtype="step", lw=1.3,
                              color=colour, label=f"{hour:g} h")
            axis.set_title(f"{name} synapses")
            axis.set_xlabel("Weight")
            axis.set_ylabel("Number of synapses")

    for unused in range(num_panels, flat_axes.size):
        flat_axes[unused].axis("off")

    fig.suptitle(title)
    _save(fig, out_path)
