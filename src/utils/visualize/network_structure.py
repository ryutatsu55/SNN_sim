"""疎な (COO) ネットワーク構造の可視化。

N が数万・シナプスが数千万規模になると、既存の `visualize.network` (辺ごとに
`ax.annotate` を呼ぶ Python ループ) も密な重み行列の `imshow` も使えない。
本モジュールは COO (row, col) と 1D の重み/遅延配列だけを受け取り、サンプリングと
粗視化で「見て意味のある」図に落とす。

いずれの関数も E/I の分類は `layout.ids_by_mode()` から得る。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial.distance import cdist

# 送信種別 × 受信種別のブロック名と描画色
BLOCK_ORDER = ("EE", "EI", "IE", "II")
BLOCK_COLORS = {
    "EE": "tab:red",
    "EI": "tab:orange",
    "IE": "tab:blue",
    "II": "tab:purple",
}


def excitatory_flags(layout, total_neurons: int) -> np.ndarray:
    """グローバルID -> 興奮性なら True の bool 配列を返す。"""
    ids = layout.ids_by_mode()
    flags = np.zeros(total_neurons, dtype=bool)
    flags[np.asarray(ids.get("excitatory", []), dtype=np.int64)] = True
    return flags


def block_masks(row: np.ndarray, col: np.ndarray, is_exc: np.ndarray) -> dict[str, np.ndarray]:
    """各シナプスが EE / EI / IE / II のどれかを示すブールマスクを返す。"""
    src_exc = is_exc[np.asarray(row, dtype=np.int64)]
    tgt_exc = is_exc[np.asarray(col, dtype=np.int64)]
    return {
        "EE": src_exc & tgt_exc,
        "EI": src_exc & ~tgt_exc,
        "IE": ~src_exc & tgt_exc,
        "II": ~src_exc & ~tgt_exc,
    }


def display_rank(layout, total_neurons: int) -> tuple[np.ndarray, int]:
    """グローバルID -> 表示順位 (興奮性が先頭ブロック) の写像と、興奮性の数を返す。"""
    ids = layout.ids_by_mode()
    exc = np.sort(np.asarray(ids.get("excitatory", []), dtype=np.int64))
    inh = np.sort(np.asarray(ids.get("inhibitory", []), dtype=np.int64))
    rank = np.zeros(total_neurons, dtype=np.int64)
    rank[exc] = np.arange(exc.size)
    rank[inh] = np.arange(exc.size, exc.size + inh.size)
    return rank, int(exc.size)


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_network_sample(
    coords: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    layout,
    out_path: Path,
    title: str = "Network (sampled)",
    n_sample: int = 500,
    max_edges: int = 4000,
    seed: int = 0,
) -> None:
    """ニューロンを部分抽出し、その内部の結合だけを描く空間ネットワーク図。

    全ニューロン・全シナプスを描くのは大規模では不可能なので、`n_sample` 個の
    ニューロンを無作為抽出し、両端が抽出集合に入る辺だけを `LineCollection` で描く。
    辺が多すぎる場合はさらに `max_edges` 本へ間引く。
    """
    coords = np.asarray(coords, dtype=np.float64)
    total = coords.shape[0]
    rng = np.random.default_rng(seed)
    is_exc = excitatory_flags(layout, total)

    sample = np.sort(rng.choice(total, size=min(n_sample, total), replace=False))
    in_sample = np.zeros(total, dtype=bool)
    in_sample[sample] = True

    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    keep = in_sample[row] & in_sample[col]
    sub_row = row[keep]
    sub_col = col[keep]
    if sub_row.size > max_edges:
        pick = rng.choice(sub_row.size, size=max_edges, replace=False)
        sub_row = sub_row[pick]
        sub_col = sub_col[pick]

    fig, ax = plt.subplots(figsize=(7, 7))
    if sub_row.size:
        segments = np.stack([coords[sub_row, :2], coords[sub_col, :2]], axis=1)
        colors = np.where(is_exc[sub_row], "tab:red", "tab:blue")
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.25, alpha=0.25))

    ax.scatter(coords[sample][is_exc[sample], 0], coords[sample][is_exc[sample], 1],
               s=8, color="tab:red", label="excitatory", zorder=3)
    ax.scatter(coords[sample][~is_exc[sample], 0], coords[sample][~is_exc[sample], 1],
               s=8, color="tab:blue", label="inhibitory", zorder=3)

    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_aspect("equal")
    ax.set_title(f"{title}\n{sample.size} neurons sampled, {sub_row.size} edges drawn")
    ax.legend(fontsize=8, markerscale=1.5)
    _save(fig, out_path)


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
    flat_axes[0].set_yscale("log")
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
            axis.set_yscale("log")

    for unused in range(num_panels, flat_axes.size):
        flat_axes[unused].axis("off")

    fig.suptitle(title)
    _save(fig, out_path)
