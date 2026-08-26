"""ネットワークを**空間に置いたグラフ**として描く。

このモジュールが担うのは「細胞体がどこにあり、どれとどれが繋がっているか」の 2 枚だけ:

- `network` : 結合を細胞体どうしを結ぶ直線の矢印として描く。ニューロンとエッジを
  サンプリングするので大規模でも破綻しない。
- `axon_network` : 同じ配置を、結合を**軸索の折れ線**として描いたもの (`axon_growth` 専用)。
  `network` と同じ seed から同じ順に乱数を引くので、2 枚は同じニューロン・同じ結合を映す。

座標を持たない図はここには無い。結合構造そのものの図 (粗視化した結合密度・距離依存の
結合確率) は `matrices.py`、値の分布 (重み・遅延) は `distributions.py`。

入力は COO (row, col と index 整合の 1D 配列)。密な (N, N) は受け取らない —
ビルド以降の受け渡しは COO 一本、という全体の規約 (`NetworkBuilder.global_coo()`) に従う。
E/I の分類は `layout.ids_by("polarity")` から得る。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from src.utils.plotting.area import draw_area

# 送信元の極性で決まるエッジ色 (ノードの tab:red / tab:blue に合わせた半透明版)
EDGE_COLORS = {True: (0.8, 0.2, 0.2, 0.5), False: (0.2, 0.2, 0.8, 0.5)}
EDGE_COLOR_NO_LAYOUT = (0.4, 0.4, 0.4, 0.5)


def _sample_nodes(total: int, n_sample: int, rng) -> np.ndarray:
    """描画するニューロンを選ぶ。`network` と `axon_network` が**同じ列を消費する**ので、
    同じ seed なら 2 枚の図に出るニューロンは一致する。"""
    return np.sort(rng.choice(total, size=min(n_sample, total), replace=False))


def _excitatory_mask(layout, total: int):
    """グローバル ID -> 興奮性か。layout が無ければ None (色分けしない)。"""
    if layout is None:
        return None
    exc_ids = np.asarray(layout.ids_by("polarity").get("excitatory", []), dtype=np.int64)
    is_exc = np.zeros(total, dtype=bool)
    is_exc[exc_ids] = True
    return is_exc


def _draw_nodes(ax, x, y, sample, is_exc, node_size) -> None:
    """細胞体を描く。E/I が分かるなら赤/青、分からなければ一色。"""
    if is_exc is None:
        ax.scatter(x[sample], y[sample], s=node_size, color='darkgray',
                   edgecolors='black', zorder=3)
        return
    exc_sample = sample[is_exc[sample]]
    inh_sample = sample[~is_exc[sample]]
    ax.scatter(x[exc_sample], y[exc_sample], s=node_size, color='tab:red',
               edgecolors='black', zorder=3, label='excitatory')
    ax.scatter(x[inh_sample], y[inh_sample], s=node_size, color='tab:blue',
               edgecolors='black', zorder=3, label='inhibitory')
    ax.legend(fontsize=9, markerscale=1.5)


def _apply_axis_limits(ax, config, area_drawn: bool) -> None:
    """軸範囲の優先順位:
      1. エリアの境界箱 (draw_area が set_limits で設定済み)。領域が図の外に切れない
      2. x_range/y_range を持つ矩形空間
      3. 座標からの自動スケール (random_circle_2d など、どちらも持たない場合)
    space: area_uniform は x_range を持たないので、エリアを渡さないと 3 に落ちる。
    """
    if area_drawn:
        return
    space_cfg = config.network.space
    x_range = getattr(space_cfg, "x_range", None)
    y_range = getattr(space_cfg, "y_range", None)
    if x_range is not None and y_range is not None:
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
    else:
        ax.margins(0.05)


def network(row: np.ndarray, col: np.ndarray, weights: np.ndarray, coords: np.ndarray,
            config, layout=None, node_size=10,
            title="network", save_path=".", n_sample=500, max_edges=4000, seed=0,
            area=None):
    """
    ニューロンの空間配置と結合 (COO) からネットワーク構造を可視化する。

    空間ネットワーク図を担う唯一の関数。大規模ネットワークでも破綻しないよう、
    ニューロンを n_sample 個サンプリングし、両端がサンプルに含まれる結合だけを
    描画する (さらに max_edges 本へ間引く)。エッジは矢印付きで、重みの符号で色分け
    (正=興奮性=赤 / 負=抑制性=青)、絶対値に応じて線の太さを変える。

    Parameters:
        row, col (np.ndarray): 各結合の送信/受信グローバルID (1D, 行優先ソート済み)。
        weights (np.ndarray): 各結合の重み (1D, row/col と index 整合)。
        coords (np.ndarray): ニューロンの座標配列。形状は (N, 3)。
        config: AppConfig。矩形空間 (x_range/y_range) なら軸範囲に使う。
        layout: NetworkLayout。渡すとノードを E/I で色分けする (省略時は一色)。
        node_size (int): ニューロン(ノード)の描画サイズ。
        title (str): グラフ/ファイル名。
        save_path (str): 画像の保存先ディレクトリ。
        n_sample (int): 描画に用いるニューロンのサンプリング数。
        max_edges (int): 描画するエッジ数の上限 (annotate ループを抑える)。
        seed (int): サンプリングの乱数シード。
        area: BaseArea。渡すと領域の境界線を背景に敷き、軸範囲もそこから取る。
    """
    coords = np.asarray(coords)
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    weights = np.asarray(weights).reshape(-1)
    if not (row.size == col.size == weights.size):
        raise ValueError("row / col / weights の長さが一致しません。")
    N = coords.shape[0]
    rng = np.random.default_rng(seed)

    # Z軸が存在する場合でも、今回は2D平面(X, Y)への投影として扱う
    x = coords[:, 0]
    y = coords[:, 1]

    # --- ニューロンのサンプリングと E/I 判定 ---
    sample = _sample_nodes(N, n_sample, rng)
    in_sample = np.zeros(N, dtype=bool)
    in_sample[sample] = True
    is_exc = _excitatory_mask(layout, N)

    fig, ax = plt.subplots(figsize=(12, 10))

    # --- 領域の境界線を背景に敷く (ノードは zorder=3、エッジは 1 なので下に回る) ---
    # 塗りは入れない。この図の主役はグラフで、part ごとの塗り分けはエッジと色が競合して
    # 読みにくくなる。領域そのものを見たいときは plot_area の図を見る。
    area_drawn = draw_area(ax, area, fill=False, boundary=True, zorder=0)

    # --- ノード描画 (layout があれば E/I で色分け) ---
    _draw_nodes(ax, x, y, sample, is_exc, node_size)

    # 描画用のスケール計算（太さの正規化用）
    abs_max = np.max(np.abs(weights)) if weights.size else 0.0
    max_weight = abs_max if abs_max > 0 else 1.0
    # 結合（エッジ）を抽出し、両端がサンプルに含まれるものだけ残す。
    # 重み 0 の結合を落とすのは「線幅 0 の矢印を描かない」ため。COO は実結合しか
    # 持たないが、可塑性で 0 まで落ちた結合はここに含まれる。
    keep = (np.abs(weights) != 0) & in_sample[row] & in_sample[col]
    sources, targets, edge_w = row[keep], col[keep], weights[keep]
    # エッジが多すぎる場合はさらに max_edges 本へ間引く (annotate は1本ずつ描くため)
    if sources.size > max_edges:
        pick = rng.choice(sources.size, size=max_edges, replace=False)
        sources, targets, edge_w = sources[pick], targets[pick], edge_w[pick]

    # 矢印がノードの中心に刺さるのを防ぐためのマージン計算
    # (scatterの s は面積なので、半径は平方根に比例)
    node_margin = np.sqrt(node_size) * 0.8

    for s, t, w in zip(sources, targets, edge_w):

        # 重みの強さに応じて線の太さを変更 (最大2.0)
        lw = (abs(w) / max_weight) * 2.0

        # エッジ色は出力元ノード (source) の色に揃える。
        # layout があれば興奮性=赤 / 抑制性=青、無ければノードと同じ灰色。
        color = EDGE_COLOR_NO_LAYOUT if is_exc is None else EDGE_COLORS[bool(is_exc[s])]

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
    _apply_axis_limits(ax, config, area_drawn)

    plt.tight_layout()

    plt.savefig(f"{save_path}/{title}.png", dpi=300, bbox_inches='tight')
    print(f"Network visualization saved to {save_path}/{title}.png")

    plt.close()


def _axon_polyline(geometry, neuron: int) -> np.ndarray:
    """ニューロン 1 本の軸索を頂点列 (V, 2) にする。軸索が無ければ空配列。

    セグメントは端点を共有して連なっている (`seg_start[k+1] == seg_end[k]`) ので、
    始点を並べて最後に終端を足せば折れ線になる。
    """
    lo, hi = int(geometry.offsets[neuron]), int(geometry.offsets[neuron + 1])
    if hi <= lo:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack([geometry.seg_start[lo:hi], geometry.seg_end[hi - 1]])


def _contact_polyline(geometry, edge: int) -> tuple[np.ndarray, np.ndarray]:
    """シナプス 1 本ぶんの「細胞体から接触点まで」の軸索経路と、その接触点。

    接触したセグメントの途中で折り返すので、`_axon_polyline` のように軸索全体を
    使うのではなく、接触セグメントの始点まで並べてから接触点を足す。
    """
    pre = int(geometry.pre[edge])
    seg = int(geometry.contact_seg[edge])
    t = float(geometry.contact_t[edge])
    a, b = geometry.seg_start[seg], geometry.seg_end[seg]
    contact = a + t * (b - a)
    return np.vstack([geometry.seg_start[int(geometry.offsets[pre]):seg + 1], contact]), contact


def axon_network(geometry, coords, config, layout=None, node_size=10, title="axon_network",
                 save_path=".", n_sample=500, max_edges=4000, seed=0, area=None,
                 show_axons=True):
    """結合を**軸索伸長過程の折れ線**として描く (直線矢印で描く `network` の対になる図)。

    `axon_growth` では結合は細胞体を結ぶ直線ではなく、伸びた軸索が誰かの樹状突起円を
    横切った結果として生まれる。この図はその経路をそのまま描くので、
    「どのブリッジを通ってモジュール間がつながったのか」が読める。

    `network()` と**同じ seed から同じ順に乱数を引く**ので、2 枚の図に出るニューロンと
    結合は一致する (エッジの順序も、COO の行優先順で揃う)。
    並べて比較するための性質なので、サンプリングの手順を変えるときは両方同時に変えること。

    重みは受け取らない。この図の主題は経路の形なので線幅は一定。
    矢印も描かない: 実線 = pre から伸びた軸索、破線 = 接触点から post 細胞体へ届いた
    樹状突起、という区別がそのまま向きを表すし、annotate を数千回呼ぶと折れ線では重すぎる。

    Parameters:
        geometry: `seg_start` / `seg_end` / `offsets` / `pre` / `post` / `contact_seg` /
            `contact_t` を持つオブジェクト (`AxonGrowthTopology.axon_geometry()` の戻り値)。
            ダックタイピングで受けるので `src/models` には依存しない。
        coords (np.ndarray): ニューロンの座標配列 (N, 2 以上)。
        config: AppConfig。矩形空間 (x_range/y_range) なら軸範囲に使う。
        layout: NetworkLayout。渡すと E/I で色分けする (省略時は一色)。
        n_sample (int): 描画に用いるニューロンのサンプリング数。
        max_edges (int): 描画する結合数の上限。
        seed (int): サンプリングの乱数シード。`network()` と揃えること。
        area: BaseArea。渡すと領域の境界線を背景に敷き、軸範囲もそこから取る。
        show_axons (bool): サンプルしたニューロンの軸索**全体**を薄いグレーで下敷きにする。
            結合を作らなかった軸索もここに出るので、伸長過程そのものが見える。
    """
    coords = np.asarray(coords, dtype=np.float64)
    N = coords.shape[0]
    rng = np.random.default_rng(seed)
    x, y = coords[:, 0], coords[:, 1]

    # --- network() と同一のサンプリング (乱数の消費順まで同じ) ---
    sample = _sample_nodes(N, n_sample, rng)
    in_sample = np.zeros(N, dtype=bool)
    in_sample[sample] = True
    is_exc = _excitatory_mask(layout, N)

    pre = np.asarray(geometry.pre, dtype=np.int64)
    post = np.asarray(geometry.post, dtype=np.int64)
    edges = np.nonzero(in_sample[pre] & in_sample[post])[0]
    if edges.size > max_edges:
        edges = edges[rng.choice(edges.size, size=max_edges, replace=False)]

    fig, ax = plt.subplots(figsize=(12, 10))

    area_drawn = draw_area(ax, area, fill=False, boundary=True, zorder=0)

    # --- 下敷き: サンプルしたニューロンの軸索を丸ごと ---
    underlay = []
    if show_axons:
        underlay = [poly for poly in (_axon_polyline(geometry, int(i)) for i in sample)
                    if len(poly) > 1]
        if underlay:
            ax.add_collection(LineCollection(underlay, colors="0.45", linewidths=0.6,
                                             alpha=0.25, zorder=0.5))

    # --- 結合を作った経路 (実線) と、接触点から細胞体まで (破線) ---
    paths, stubs, colors = [], [], []
    for edge in edges:
        path, contact = _contact_polyline(geometry, int(edge))
        paths.append(path)
        stubs.append(np.vstack([contact, coords[post[edge], :2]]))
        colors.append(EDGE_COLOR_NO_LAYOUT if is_exc is None
                      else EDGE_COLORS[bool(is_exc[pre[edge]])])
    if paths:
        ax.add_collection(LineCollection(paths, colors=colors, linewidths=1.0, zorder=1))
        ax.add_collection(LineCollection(stubs, colors=colors, linewidths=0.8,
                                         linestyles=(0, (2, 2)), zorder=1.5))

    _draw_nodes(ax, x, y, sample, is_exc, node_size)

    # 凡例は _draw_nodes が付けた E/I に線種の説明を足す (色は送信元の極性で決まる)。
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.4", lw=1.2, label="axon (made a synapse)"))
    handles.append(Line2D([], [], color="0.4", lw=1.0, ls=(0, (2, 2)),
                          label="dendrite reach"))
    if underlay:
        handles.append(Line2D([], [], color="0.45", lw=0.6, alpha=0.6, label="all axons"))
    ax.legend(handles=handles, fontsize=9, markerscale=1.5)

    ax.set_aspect('equal')
    ax.set_title(f"{title}\n{sample.size} neurons sampled, {len(paths)} synapses "
                 f"drawn along axons, {len(underlay)} axons")
    ax.set_xlabel("X Coordinate [um]")
    ax.set_ylabel("Y Coordinate [um]")
    _apply_axis_limits(ax, config, area_drawn)

    plt.tight_layout()
    plt.savefig(f"{save_path}/{title}.png", dpi=300, bbox_inches='tight')
    print(f"Axon network visualization saved to {save_path}/{title}.png")
    plt.close()
