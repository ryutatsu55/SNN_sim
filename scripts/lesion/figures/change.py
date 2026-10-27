"""2 つの記録窓のあいだの**変化量** (後 − 先) を描く。run 全体で 1 枚ずつ。

比べる窓の組は 2 種類:

- **回復** (`切断直後 → TARGET_HOUR`): 切断後の可塑性と活動だけによる変化。
  - `weight_change_matrix` : 重み行列の差 (`weight_matrix` の差分版)
  - `weight_change_network` : 重みの差を空間に置いたもの (`weight_network` の差分版)
  - `weight_change_axon_network` : 重みの差を軸索の折れ線で描いたもの (`axon_network` の差分版)
  - `correlation_change_matrix` : 相関行列の差 (`correlation_matrix` の差分版)
  - `correlation_change_network` : 相関の差を空間に置いたもの (`correlation_network` の差分版)
- **切断の前後** (`切断前 → 切断直後`): 切断そのものが何を変えたか。
  - `weight_change_matrix_across_cut` / `weight_change_network_across_cut` /
    `weight_change_axon_network_across_cut` / `correlation_change_matrix_across_cut` /
    `correlation_change_network_across_cut` (上の同名の図と同じ描き方)
- **結合ごとの散布図**: 切断後に残った結合 1 本 = 1 点で、縦か横が回復の重みの差。点の色は
  pre / post の極性 (E→E / E→I / I→E / I→I)。
  - `weight_change_vs_correlation_change` : 横 = 切断の前後の相関の差
  - `weight_change_vs_correlation` : 横 = 切断直後の相関
  - `pre_rate_vs_weight_change` / `post_rate_vs_weight_change` : 縦 = `TARGET_HOUR` の窓の
    pre / post の発火率

  相関・発火率の軸には、群ごとと全体の平均を破線で引く。ただし相関の差の図は差の平均ではなく、
  切断直後の平均相関を凡例に書く。

切断前は baseline の窓 (親 run の記録)、切断直後は最初の post の窓 (0 h の probe)、
`TARGET_HOUR` はその時刻に最も近い post の窓。**切断直後の窓は settle の後なので、
切断の前後の差には settle 中の可塑性による変化も含まれる。**

**重みの差は、切断前の窓を含むときは切断前の結合の上で取る。** 切ったシナプスは切断後の
重みを 0 として数えるので、「切断前の重みぶんの減少」として現れる。切断前と切断後の重みは
本数が違うので、(pre, post) で引き当てる (`src/utils/analysis/weights.align_subset_to_coo`)。
位置で対応づけてはいけない。回復の差は両端とも切断後の結合なので、そのまま引く。
軸索で描く図も同じで、切断前の窓を含むときは切断前の幾何 (`Series.geometry_pre()`) の上に描く。

**色の範囲は図ごとにデータから決める。** run をまたいで、また図をまたいで色を比べることは
できない (範囲はカラーバーで読む)。

1 ファイルにまとめているのは、どの図も同じ窓の選び方 (`recovery_windows` /
`cut_windows`) を使うから —— 片方だけ変えると、同じ「20 h」が図ごとに別の窓を指してしまう。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from src.utils.analysis.correlation import RATE_BIN_MS, neuronal_correlation
from src.utils.analysis.spikes import firing_rates
from src.utils.analysis.weights import align_subset_to_coo
from src.utils.runview import MissingData, optional
from scripts.lesion.figures import save, style
from scripts.lesion.figures.area import draw_area
from scripts.lesion.figures.pair_scatter import draw_block_scatter
from scripts.lesion.figures.network import (MAX_EDGE_WIDTH, MAX_EDGES, NODE_SIZE, SAMPLE_SEED,
                                            N_SAMPLE, SHOW_ALL_AXONS, apply_axis_limits,
                                            axon_polyline, contact_polyline, draw_edges,
                                            draw_nodes, excitatory_mask, sample_nodes,
                                            save_legend_figure)
from scripts.lesion.figures.weight_matrix import densify

# 回復を測る先の時刻 [h] (切断からの経過)。**この時刻の post 窓が無い run では回復の図は出ない。**
TARGET_HOUR = 20.0
# `TARGET_HOUR` と窓の時刻のずれの許容 [h]。窓の時刻は ms から h へ直した float なので、
# 完全一致では比べない。
HOUR_TOLERANCE = 0.25

# 並べ替え軸。**重み行列・相関行列と同じ。** 差分の図も同じブロック配置で読めるように。
ORDER_AXES = style.ORDER_AXES

# 見た目。増加が赤、減少が青。0 (変化なし・結合なし) は白。
CMAP = "RdBu_r"
# 相関が未定義 (どちらかの窓で 1 発も撃たないニューロン) の色。**変化 0 とは別物。**
UNDEFINED_COLOR = "0.6"
MATRIX_FIGSIZE = (7, 6)
NETWORK_FIGSIZE = (12, 10)
MATRIX_DPI = 200
NETWORK_DPI = 300
# 最外ブロックの名前を目盛りに出す上限。これを超えると文字が潰れるので番号のままにする。
MAX_BLOCK_TICKS = 16
# `correlation_change_network` で線にする変化の大きさの下限。**窓をまたいで比べるあいだは
# 動かさないこと** (`correlation_network` の閾値と同じ理由)。
MIN_CORRELATION_CHANGE = 0.2
CORRELATION_EDGE_WIDTH = 0.5
# `weight_change_axon_network` の破線 (接触点から post の細胞体) の太さ。
STUB_WIDTH = 0.5


# ======================================================================================
# 窓の選び方
# ======================================================================================

def _baseline_window(series):
    baseline = [window for window in series.windows if window.is_baseline]
    if not baseline:
        raise MissingData("baseline window", "切断前の窓がありません")
    return baseline[-1]


def _post_windows(series):
    post = [window for window in series.windows if not window.is_baseline]
    if not post:
        raise MissingData("post window", "切断後の窓がありません")
    return post


def recovery_windows(series):
    """(切断直後, `TARGET_HOUR`) の 2 窓を返す。

    切断直後は最初の post の窓、後者は `TARGET_HOUR` に最も近い post の窓。
    どちらかが無ければ `MissingData`。
    """
    post = _post_windows(series)
    target = min(post, key=lambda window: abs(window.hour - TARGET_HOUR))
    if abs(target.hour - TARGET_HOUR) > HOUR_TOLERANCE:
        raise MissingData(f"{TARGET_HOUR:g} h window",
                          f"{TARGET_HOUR:g} h の窓がありません (最も近い post 窓は"
                          f" {target.hour:+.2f} h)")
    return post[0], target


def cut_windows(series):
    """(切断前, 切断直後) の 2 窓を返す。切断前は最後の baseline の窓。"""
    return _baseline_window(series), _post_windows(series)[0]


def _pair_title(reference, target) -> str:
    return f"{target.label}\nminus {reference.label}"


def _pair_line(reference, target) -> str:
    return f"{target.label} minus {reference.label}"


def _symmetric_limit(values) -> float:
    """0 を中心にした色の範囲の上端。全部 0 (か空) なら 1。"""
    values = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    values = values[np.isfinite(values)]
    top = float(values.max()) if values.size else 0.0
    return top if top > 0 else 1.0


def _label_blocks(ax, layout, ordering, size: int) -> None:
    """最外ブロック (module) の境界線と名前を引く。重み行列・相関行列と同じ約束。"""
    if not ordering.enabled:
        return
    style.draw_block_boundaries(ax, ordering, size, skip=("polarity",))
    blocks = style.outer_block_labels(layout, ordering, size)
    if blocks and len(blocks) <= MAX_BLOCK_TICKS:
        ticks = [centre - 0.5 for centre, _ in blocks]
        names = [name for _, name in blocks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(ticks)
        ax.set_yticklabels(names)
    else:
        ticks = style.block_ticks(ordering, size)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)


# ======================================================================================
# 重み
# ======================================================================================

def _weight_change(series, reference, target):
    """`target` − `reference` の重みの差と、それが並ぶ結合 `(wiring, delta)` を返す。

    どちらかが切断前の窓なら切断前の結合の上で (切ったシナプスは切断後を 0)、
    そうでなければ切断後の結合の上で取る。
    """
    if not (reference.is_baseline or target.is_baseline):
        delta = (np.asarray(target.weights(), dtype=np.float64)
                 - np.asarray(reference.weights(), dtype=np.float64))
        return series.wiring(), delta

    pre, post = series.wiring_pre(), series.wiring()
    # 切断後のシナプスが切断前の何番目か。(pre, post) で引き当てる。
    post_in_pre = align_subset_to_coo(pre.row, pre.col, np.arange(pre.row.size),
                                      post.row, post.col, int(pre.shape[1])).astype(np.int64)

    def on_pre(window) -> np.ndarray:
        if window.is_baseline:
            return np.asarray(window.weights(), dtype=np.float64)
        values = np.zeros(pre.row.size, dtype=np.float64)
        values[post_in_pre] = window.weights()
        return values

    return pre, on_pre(target) - on_pre(reference)


def _draw_weight_change_matrix(series, out_path: Path, reference, target) -> None:
    wiring, delta = _weight_change(series, reference, target)
    layout = series.layout
    ordering = style.resolve_ordering(layout, style.available_order_axes(layout, ORDER_AXES))
    limit = _symmetric_limit(delta)

    fig, ax = plt.subplots(figsize=MATRIX_FIGSIZE, constrained_layout=True)
    ordered = ordering.apply(densify(wiring.row, wiring.col, delta, layout.total_neurons))
    image = ax.imshow(ordered, origin="upper", interpolation="nearest",
                      vmin=-limit, vmax=limit, cmap=CMAP)
    _label_blocks(ax, layout, ordering, ordered.shape[0])
    ax.set_title(f"Weight change\n{_pair_title(reference, target)}", fontsize=10)
    ax.set_xlabel("post neuron id")
    ax.set_ylabel("pre neuron id")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="weight change")
    save(fig, out_path, tight_layout=False, dpi=MATRIX_DPI)


def weight_change_matrix(series, out_path: Path) -> None:
    """重み行列の差 (`TARGET_HOUR` − 切断直後) を描く。

    結合の無い位置も 0 (白) で塗られるので、白は「変化なし」と「結合なし」の両方を指す。
    """
    _draw_weight_change_matrix(series, out_path, *recovery_windows(series))


def weight_change_matrix_across_cut(series, out_path: Path) -> None:
    """重み行列の差 (切断直後 − 切断前) を描く。切ったシナプスは切断前の重みぶんの減少。"""
    _draw_weight_change_matrix(series, out_path, *cut_windows(series))


def _sampled_changes(row, col, delta, total: int):
    """`weight_network` と同じニューロンのサンプルと、その中で描く変化した結合の添字を返す。

    乱数の消費順は `weight_network` と同じ。添字は `|差|` の小さい順 (大きい変化を上に描く)。
    """
    rng = np.random.default_rng(SAMPLE_SEED)
    sample = sample_nodes(total, N_SAMPLE, rng)
    in_sample = np.zeros(total, dtype=bool)
    in_sample[sample] = True
    keep = np.flatnonzero((delta != 0) & in_sample[row] & in_sample[col])
    if keep.size > MAX_EDGES:
        keep = keep[rng.choice(keep.size, size=MAX_EDGES, replace=False)]
    return sample, keep[np.argsort(np.abs(delta[keep]), kind="stable")]


def _weight_change_geometry(series, reference, target):
    """`_weight_change` が差を取る結合 (切断前 / 切断後) の軸索幾何。"""
    if reference.is_baseline or target.is_baseline:
        return series.geometry_pre()
    return series.geometry()


def _draw_weight_change_network(series, out_path: Path, reference, target) -> None:
    wiring, delta = _weight_change(series, reference, target)
    config, layout = series.config, series.layout
    area = optional(series.area)
    coords = np.asarray(series.coords())
    x, y = coords[:, 0], coords[:, 1]
    total = coords.shape[0]
    is_exc = excitatory_mask(layout, total)
    row = np.asarray(wiring.row, dtype=np.int64)
    col = np.asarray(wiring.col, dtype=np.int64)
    limit = _symmetric_limit(delta)
    norm = plt.Normalize(-limit, limit)
    cmap = plt.get_cmap(CMAP)
    node_margin = np.sqrt(NODE_SIZE) * 0.8

    sample, keep = _sampled_changes(row, col, delta, total)
    values = delta[keep]

    fig, ax = plt.subplots(figsize=NETWORK_FIGSIZE)
    area_drawn = draw_area(ax, area, fill=False, boundary=True, zorder=0)
    draw_nodes(ax, x, y, sample, is_exc, NODE_SIZE)
    draw_edges(ax, x, y, row[keep], col[keep], np.abs(values) / limit * MAX_EDGE_WIDTH,
               is_exc, node_margin, colors=cmap(norm(values)))
    ax.set_aspect("equal")
    ax.set_title(f"Weight change network\n{_pair_title(reference, target)}\n"
                 f"{sample.size} neurons sampled, {keep.size} changed edges drawn",
                 fontsize=10)
    ax.set_xlabel("X Coordinate [um]")
    ax.set_ylabel("Y Coordinate [um]")
    apply_axis_limits(ax, config, area_drawn)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 fraction=0.025, pad=0.02, label="weight change")
    save(fig, out_path, tight_layout=False, dpi=NETWORK_DPI, bbox_inches="tight")


def weight_change_network(series, out_path: Path) -> None:
    """重みの差 (`TARGET_HOUR` − 切断直後) を空間に置いて描く (`weight_network` の差分版)。

    ニューロンのサンプリングは `weight_network` と同じ。色は差の符号と大きさ、線の太さは
    `|差| / 色の範囲の上端` に比例する。**差が 0 の結合は描かない。**
    """
    _draw_weight_change_network(series, out_path, *recovery_windows(series))


def weight_change_network_across_cut(series, out_path: Path) -> None:
    """重みの差 (切断直後 − 切断前) を `weight_change_network` と同じ描き方で描く。

    切ったシナプスは切断前の重みぶんの減少 (青) として現れる。
    """
    _draw_weight_change_network(series, out_path, *cut_windows(series))


def _draw_weight_change_axon_network(series, out_path: Path, reference, target) -> None:
    wiring, delta = _weight_change(series, reference, target)
    geometry = _weight_change_geometry(series, reference, target)
    config, layout = series.config, series.layout
    area = optional(series.area)
    coords = np.asarray(series.coords(), dtype=np.float64)
    x, y = coords[:, 0], coords[:, 1]
    total = coords.shape[0]
    is_exc = excitatory_mask(layout, total)
    pre = np.asarray(geometry.pre, dtype=np.int64)
    post = np.asarray(geometry.post, dtype=np.int64)
    # 差を幾何のシナプスの並びへ。**位置ではなく (pre, post) で引き当てる。**
    delta = align_subset_to_coo(wiring.row, wiring.col, delta, pre, post, total)
    limit = _symmetric_limit(delta)
    norm = plt.Normalize(-limit, limit)
    cmap = plt.get_cmap(CMAP)

    sample, keep = _sampled_changes(pre, post, delta, total)
    values = delta[keep]
    colors = cmap(norm(values))
    widths = np.abs(values) / limit * MAX_EDGE_WIDTH

    fig, ax = plt.subplots(figsize=NETWORK_FIGSIZE)
    area_drawn = draw_area(ax, area, fill=False, boundary=True, zorder=0)
    underlay = []
    if SHOW_ALL_AXONS:
        underlay = [poly for poly in (axon_polyline(geometry, int(i)) for i in sample)
                    if len(poly) > 1]
        if underlay:
            ax.add_collection(LineCollection(underlay, colors="0.45", linewidths=0.6,
                                             alpha=0.25, zorder=0.5))
    paths, stubs = [], []
    for edge in keep:
        path, contact = contact_polyline(geometry, int(edge))
        paths.append(path)
        stubs.append(np.vstack([contact, coords[post[edge], :2]]))
    if paths:
        ax.add_collection(LineCollection(paths, colors=colors, linewidths=widths, zorder=1))
        # 破線は**一定の太さ**。破線の周期は線幅に比例して縮むので、変化の小さい
        # (細い) 線を破線にすると無数の小片になり、保存が終わらなくなる。
        ax.add_collection(LineCollection(stubs, colors=colors, linewidths=STUB_WIDTH,
                                         linestyles=(0, (2, 2)), zorder=1.5))
    draw_nodes(ax, x, y, sample, is_exc, NODE_SIZE)

    # 凡例は `axon_network` と同じく本体に載せず別ファイルへ。
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.4", lw=1.2, label="axon (synapse changed)"))
    handles.append(Line2D([], [], color="0.4", lw=1.0, ls=(0, (2, 2)),
                          label="dendrite reach"))
    if underlay:
        handles.append(Line2D([], [], color="0.45", lw=0.6, alpha=0.6, label="all axons"))
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.set_aspect("equal")
    ax.set_title(f"Weight change axon network\n{_pair_title(reference, target)}\n"
                 f"{sample.size} neurons sampled, {len(paths)} changed synapses drawn",
                 fontsize=10)
    ax.set_xlabel("X Coordinate [um]")
    ax.set_ylabel("Y Coordinate [um]")
    apply_axis_limits(ax, config, area_drawn)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 fraction=0.025, pad=0.02, label="weight change")
    out_path = Path(out_path)
    save(fig, out_path, tight_layout=False, dpi=NETWORK_DPI, bbox_inches="tight")
    save_legend_figure(handles, out_path.with_name(f"{out_path.stem}_legend{out_path.suffix}"))


def weight_change_axon_network(series, out_path: Path) -> None:
    """重みの差 (`TARGET_HOUR` − 切断直後) を軸索の折れ線で描く (`axon_network` の差分版)。

    描く結合の選び方・色・太さは `weight_change_network` と同じ。線の様式 (実線 = 軸索、
    破線 = 接触点から post の細胞体) と凡例を別ファイルに出すことは `axon_network` と同じ。
    軸索の幾何を持たない run では `series.geometry()` が `MissingData` を投げる。
    """
    _draw_weight_change_axon_network(series, out_path, *recovery_windows(series))


def weight_change_axon_network_across_cut(series, out_path: Path) -> None:
    """重みの差 (切断直後 − 切断前) を `weight_change_axon_network` と同じ描き方で描く。

    「どの軸索のどこで切ったか」が読める。軸索は切断前の幾何 (`series.geometry_pre()`)
    から描くので、切ったシナプスの接触点も残っている。
    """
    _draw_weight_change_axon_network(series, out_path, *cut_windows(series))


# ======================================================================================
# 相関
# ======================================================================================

def _correlation(series, window) -> np.ndarray:
    """窓の (N, N) の相関行列。未定義のペアは nan。"""
    spikes = window.spikes()
    return neuronal_correlation(spikes.times, spikes.ids, int(series.total_neurons),
                                window.record_window_ms)


def _correlation_change(series, reference, target) -> np.ndarray:
    """`target` − `reference` の相関行列の差。どちらかの窓で未定義のペアは nan。"""
    return _correlation(series, target) - _correlation(series, reference)


def _draw_correlation_change_matrix(series, out_path: Path, reference, target) -> None:
    delta = _correlation_change(series, reference, target)
    layout = series.layout
    ordering = style.resolve_ordering(layout, style.available_order_axes(layout, ORDER_AXES))
    limit = _symmetric_limit(delta)
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(UNDEFINED_COLOR)

    fig, ax = plt.subplots(figsize=MATRIX_FIGSIZE, constrained_layout=True)
    ordered = ordering.apply(delta)
    image = ax.imshow(ordered, origin="upper", interpolation="nearest",
                      vmin=-limit, vmax=limit, cmap=cmap)
    _label_blocks(ax, layout, ordering, ordered.shape[0])
    ax.set_title(f"Correlation change ({RATE_BIN_MS:g} ms bins)\n"
                 f"{_pair_title(reference, target)}", fontsize=10)
    ax.set_xlabel("neuron id")
    ax.set_ylabel("neuron id")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="correlation change")
    save(fig, out_path, tight_layout=False, dpi=MATRIX_DPI)


def correlation_change_matrix(series, out_path: Path) -> None:
    """相関行列の差 (`TARGET_HOUR` − 切断直後) を描く。

    **符号付きのまま描く** (`correlation_matrix` と違って負を潰さない)。どちらかの窓で
    未定義のペアは `UNDEFINED_COLOR`。
    """
    _draw_correlation_change_matrix(series, out_path, *recovery_windows(series))


def correlation_change_matrix_across_cut(series, out_path: Path) -> None:
    """相関行列の差 (切断直後 − 切断前) を `correlation_change_matrix` と同じ描き方で描く。"""
    _draw_correlation_change_matrix(series, out_path, *cut_windows(series))


def _draw_correlation_change_network(series, out_path: Path, reference, target) -> None:
    delta = _correlation_change(series, reference, target)
    config, layout = series.config, series.layout
    area = optional(series.area)
    coords = np.asarray(series.coords())
    x, y = coords[:, 0], coords[:, 1]
    total = coords.shape[0]
    is_exc = excitatory_mask(layout, total)
    limit = _symmetric_limit(delta)
    norm = plt.Normalize(-limit, limit)

    # 無向なので上三角だけ。`nan` (未定義) は比較で落ちる。
    rows, cols = np.nonzero(np.triu(np.abs(delta) >= MIN_CORRELATION_CHANGE, 1))
    values = delta[rows, cols]
    order = np.argsort(np.abs(values), kind="stable")
    rows, cols, values = rows[order], cols[order], values[order]

    fig, ax = plt.subplots(figsize=NETWORK_FIGSIZE)
    area_drawn = draw_area(ax, area, fill=False, boundary=True, zorder=0)
    segments = np.stack([np.column_stack([x[rows], y[rows]]),
                         np.column_stack([x[cols], y[cols]])], axis=1)
    ax.add_collection(LineCollection(segments, array=values, cmap=CMAP, norm=norm,
                                     linewidths=CORRELATION_EDGE_WIDTH, zorder=1))
    draw_nodes(ax, x, y, np.arange(total), is_exc, NODE_SIZE)
    ax.set_aspect("equal")
    ax.set_title(f"Correlation change network ({RATE_BIN_MS:g} ms bins)\n"
                 f"{_pair_title(reference, target)}\n"
                 f"{values.size} pairs with |change| >= {MIN_CORRELATION_CHANGE:g}",
                 fontsize=10)
    ax.set_xlabel("X Coordinate [um]")
    ax.set_ylabel("Y Coordinate [um]")
    apply_axis_limits(ax, config, area_drawn)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax,
                 fraction=0.025, pad=0.02, label="correlation change")
    save(fig, out_path, tight_layout=False, dpi=NETWORK_DPI, bbox_inches="tight")


def correlation_change_network(series, out_path: Path) -> None:
    """相関の差 (`TARGET_HOUR` − 切断直後) を空間に置いて描く (`correlation_network` の差分版)。

    **描くのは `|差| >= MIN_CORRELATION_CHANGE` のペアだけ。** 無向の線で、色が差。
    ニューロンはサンプリングしない (`correlation_network` と同じ)。
    """
    _draw_correlation_change_network(series, out_path, *recovery_windows(series))


def correlation_change_network_across_cut(series, out_path: Path) -> None:
    """相関の差 (切断直後 − 切断前) を `correlation_change_network` と同じ描き方で描く。"""
    _draw_correlation_change_network(series, out_path, *cut_windows(series))


# ======================================================================================
# 結合ごとの散布図
# ======================================================================================

def _recovery_on_kept(series):
    """切断後に残った結合の `(row, col, 回復の重みの差, 切断直後の窓, TARGET_HOUR の窓)`。"""
    just_after, target = recovery_windows(series)
    wiring, delta = _weight_change(series, just_after, target)
    return (np.asarray(wiring.row, dtype=np.int64), np.asarray(wiring.col, dtype=np.int64),
            delta, just_after, target)


def weight_change_vs_correlation_change(series, out_path: Path) -> None:
    """残った結合ごとに、横 = 切断の前後の相関の差、縦 = 回復の重みの差を描く。

    相関は (pre, post) のペアの値。どちらかの窓で相関が未定義のペアは描かない。
    """
    row, col, delta, just_after, target = _recovery_on_kept(series)
    before, _ = cut_windows(series)
    at_cut = _correlation(series, just_after)
    correlation = (at_cut - _correlation(series, before))[row, col]
    draw_block_scatter(
        series.layout, out_path, row, col, correlation, delta,
        xlabel=f"correlation change ({RATE_BIN_MS:g} ms bins)", ylabel="weight change",
        title=(f"weight: {_pair_line(just_after, target)}\n"
               f"correlation: {_pair_line(before, just_after)}"),
        zero_x=True, zero_y=True, legend_means={"corr(0h)": at_cut[row, col]})


def weight_change_vs_correlation(series, out_path: Path) -> None:
    """残った結合ごとに、横 = 切断直後の相関、縦 = 回復の重みの差を描く。"""
    row, col, delta, just_after, target = _recovery_on_kept(series)
    correlation = _correlation(series, just_after)[row, col]
    draw_block_scatter(
        series.layout, out_path, row, col, correlation, delta,
        xlabel=f"correlation at {just_after.label} ({RATE_BIN_MS:g} ms bins)",
        ylabel="weight change",
        title=f"weight: {_pair_line(just_after, target)}",
        zero_y=True, mean_x="corr")


def _draw_rate_vs_weight_change(series, out_path: Path, side: str) -> None:
    row, col, delta, just_after, target = _recovery_on_kept(series)
    rates = firing_rates(target.spikes().ids, int(series.total_neurons),
                         target.record_window_ms)
    ids = row if side == "pre" else col
    draw_block_scatter(
        series.layout, out_path, row, col, delta, rates[ids],
        xlabel="weight change", ylabel=f"{side} firing rate at {target.label} [Hz]",
        title=f"weight: {_pair_line(just_after, target)}",
        zero_x=True, mean_y="rate")


def pre_rate_vs_weight_change(series, out_path: Path) -> None:
    """残った結合ごとに、横 = 回復の重みの差、縦 = `TARGET_HOUR` の窓の pre の発火率を描く。"""
    _draw_rate_vs_weight_change(series, out_path, "pre")


def post_rate_vs_weight_change(series, out_path: Path) -> None:
    """残った結合ごとに、横 = 回復の重みの差、縦 = `TARGET_HOUR` の窓の post の発火率を描く。"""
    _draw_rate_vs_weight_change(series, out_path, "post")
