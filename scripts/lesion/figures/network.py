"""ネットワークを**空間に置いたグラフ**として描く。

- `network` : 結合を細胞体どうしを結ぶ直線の矢印として描く。ニューロンとエッジを
  サンプリングするので大規模でも破綻しない。
- `axon_network` : 同じ配置を、結合を**軸索の折れ線**として描いたもの (`axon_growth` 専用)。
  `network` と同じ seed から同じ順に乱数を引くので、2 枚は同じニューロン・同じ結合を映す。

入力は COO。E/I の分類は `layout.ids_by("polarity")` から得る。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from src.utils.runview import optional
from scripts.lesion.figures import save
from scripts.lesion.figures.area import draw_area


EDGE_COLORS = {True: (0.8, 0.2, 0.2, 0.5), False: (0.2, 0.2, 0.8, 0.5)}

EDGE_COLOR_NO_LAYOUT = (0.4, 0.4, 0.4, 0.5)

# 保存時の解像度。**引数にしない** —— 変えたくなったらここを直す。
DPI = 300
NODE_SIZE = 10
# サンプリング数。**2 枚が同じ seed から同じ順に乱数を引く**ので、network と
# axon_network に同じニューロン・同じ結合が出る。片方だけ変えないこと。
N_SAMPLE = 500
MAX_EDGES = 4000
SAMPLE_SEED = 0
NETWORK_TITLE = "network_sample"
AXON_TITLE = "axon_network"
# 結合を作らなかった軸索も薄い下敷きとして描くか。
SHOW_ALL_AXONS = True


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

def network(built, out_path):
    """
    ニューロンの空間配置と結合 (COO) からネットワーク構造を可視化する。

    空間ネットワーク図を担う唯一の関数。大規模ネットワークでも破綻しないよう、
    ニューロンを N_SAMPLE 個サンプリングし、両端がサンプルに含まれる結合だけを
    描画する (さらに MAX_EDGES 本へ間引く)。エッジは矢印付きで、重みの符号で色分け
    (正=興奮性=赤 / 負=抑制性=青)、絶対値に応じて線の太さを変える。

    エリアは**あれば**境界線を背景に敷き、軸範囲もそこから取る。無くてもこの図は
    成立するので `optional()` で読む (座標は無いと描けないので `coords()` は素で呼ぶ)。
    """
    coo = built.coo()
    config, layout = built.config, built.layout
    area = optional(built.area)

    coords = np.asarray(built.coords())
    row = np.asarray(coo.row, dtype=np.int64)
    col = np.asarray(coo.col, dtype=np.int64)
    weights = np.asarray(coo.weights).reshape(-1)
    if not (row.size == col.size == weights.size):
        raise ValueError("row / col / weights の長さが一致しません。")
    N = coords.shape[0]
    rng = np.random.default_rng(SAMPLE_SEED)

    # Z軸が存在する場合でも、今回は2D平面(X, Y)への投影として扱う
    x = coords[:, 0]
    y = coords[:, 1]

    # --- ニューロンのサンプリングと E/I 判定 ---
    sample = _sample_nodes(N, N_SAMPLE, rng)
    in_sample = np.zeros(N, dtype=bool)
    in_sample[sample] = True
    is_exc = _excitatory_mask(layout, N)

    fig, ax = plt.subplots(figsize=(12, 10))

    # --- 領域の境界線を背景に敷く (ノードは zorder=3、エッジは 1 なので下に回る) ---
    # 塗りは入れない。この図の主役はグラフで、part ごとの塗り分けはエッジと色が競合して
    # 読みにくくなる。領域そのものを見たいときは plot_area の図を見る。
    area_drawn = draw_area(ax, area, fill=False, boundary=True, zorder=0)

    # --- ノード描画 (layout があれば E/I で色分け) ---
    _draw_nodes(ax, x, y, sample, is_exc, NODE_SIZE)

    # 描画用のスケール計算（太さの正規化用）
    abs_max = np.max(np.abs(weights)) if weights.size else 0.0
    max_weight = abs_max if abs_max > 0 else 1.0
    # 結合（エッジ）を抽出し、両端がサンプルに含まれるものだけ残す。
    # 重み 0 の結合を落とすのは「線幅 0 の矢印を描かない」ため。COO は実結合しか
    # 持たないが、可塑性で 0 まで落ちた結合はここに含まれる。
    keep = (np.abs(weights) != 0) & in_sample[row] & in_sample[col]
    sources, targets, edge_w = row[keep], col[keep], weights[keep]
    # エッジが多すぎる場合はさらに max_edges 本へ間引く (annotate は1本ずつ描くため)
    if sources.size > MAX_EDGES:
        pick = rng.choice(sources.size, size=MAX_EDGES, replace=False)
        sources, targets, edge_w = sources[pick], targets[pick], edge_w[pick]

    # 矢印がノードの中心に刺さるのを防ぐためのマージン計算
    # (scatterの s は面積なので、半径は平方根に比例)
    node_margin = np.sqrt(NODE_SIZE) * 0.8

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
    ax.set_title(f"{NETWORK_TITLE}\n{sample.size} neurons sampled, "
                 f"{sources.size} edges drawn")
    ax.set_xlabel("X Coordinate [um]")
    ax.set_ylabel("Y Coordinate [um]")
    _apply_axis_limits(ax, config, area_drawn)

    save(fig, out_path, dpi=DPI, bbox_inches='tight')
    print(f"Network visualization saved to {out_path}")

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

def _save_legend_figure(handles, out_path: str) -> None:
    """凡例**だけ**を別ファイルに保存する。

    軸索の図は領域いっぱいに広がるので、凡例を図の中に置くとどこに置いても
    経路の一部を隠す。図の外に出すには余白を作るしかなく、今度は絵が小さくなる。
    そこで凡例は独立した png にして、本体からは外している。
    """
    fig = plt.figure(figsize=(3.0, 0.32 * len(handles) + 0.3))
    legend = fig.legend(handles=handles, loc="center", fontsize=9, markerscale=1.5)
    # bbox_inches='tight' は「軸のある図」を前提にするため、凡例だけの図では
    # 余白が残る。凡例そのものの外接矩形を測って、そこで切り取る。
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    save(fig, out_path, tight_layout=False, dpi=DPI, bbox_inches=bbox.expanded(1.05, 1.05))

def axon_network(built, out_path):
    """結合を**軸索伸長過程の折れ線**として描く (直線矢印で描く `network` の対になる図)。

    `axon_growth` では結合は細胞体を結ぶ直線ではなく、伸びた軸索が誰かの樹状突起円を
    横切った結果として生まれる。この図はその経路をそのまま描くので、
    「どのブリッジを通ってモジュール間がつながったのか」が読める。

    `network()` と**同じ seed から同じ順に乱数を引く**ので、2 枚の図に出るニューロンと
    結合は一致する (エッジの順序も、COO の行優先順で揃う)。
    並べて比較するための性質なので、サンプリングの手順を変えるときは両方同時に変えること。

    凡例は図の中には描かず `<out_path の stem>_legend.png` として別に保存する。軸索は領域全体に
    広がるので、凡例をどこに置いても経路を隠してしまうため。

    重みは受け取らない。この図の主題は経路の形なので線幅は一定。
    矢印も描かない: 実線 = pre から伸びた軸索、破線 = 接触点から post 細胞体へ届いた
    樹状突起、という区別がそのまま向きを表すし、annotate を数千回呼ぶと折れ線では重すぎる。

    軸索の幾何を持たないコネクタ (`constant_prob` など) では `built.geometry()` が
    `MissingData` を投げるので、ここに分岐は書かない。
    """
    geometry = built.geometry()
    coords = built.coords()
    config, layout = built.config, built.layout
    area = optional(built.area)

    coords = np.asarray(coords, dtype=np.float64)
    N = coords.shape[0]
    rng = np.random.default_rng(SAMPLE_SEED)
    x, y = coords[:, 0], coords[:, 1]

    # --- network() と同一のサンプリング (乱数の消費順まで同じ) ---
    sample = _sample_nodes(N, N_SAMPLE, rng)
    in_sample = np.zeros(N, dtype=bool)
    in_sample[sample] = True
    is_exc = _excitatory_mask(layout, N)

    pre = np.asarray(geometry.pre, dtype=np.int64)
    post = np.asarray(geometry.post, dtype=np.int64)
    edges = np.nonzero(in_sample[pre] & in_sample[post])[0]
    if edges.size > MAX_EDGES:
        edges = edges[rng.choice(edges.size, size=MAX_EDGES, replace=False)]

    fig, ax = plt.subplots(figsize=(12, 10))

    area_drawn = draw_area(ax, area, fill=False, boundary=True, zorder=0)

    # --- 下敷き: サンプルしたニューロンの軸索を丸ごと ---
    underlay = []
    if SHOW_ALL_AXONS:
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

    _draw_nodes(ax, x, y, sample, is_exc, NODE_SIZE)

    # 凡例は _draw_nodes が付けた E/I に線種の説明を足す (色は送信元の極性で決まる)。
    # ただしこの図には載せず、`{title}_legend.png` として別に出す (_save_legend_figure)。
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.4", lw=1.2, label="axon (made a synapse)"))
    handles.append(Line2D([], [], color="0.4", lw=1.0, ls=(0, (2, 2)),
                          label="dendrite reach"))
    if underlay:
        handles.append(Line2D([], [], color="0.45", lw=0.6, alpha=0.6, label="all axons"))
    # _draw_nodes が ax.legend() を呼んでいるので、本体からは取り除く。
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.set_aspect('equal')
    ax.set_title(f"{AXON_TITLE}\n{sample.size} neurons sampled, "
                 f"{len(paths)} synapses drawn along axons, {len(underlay)} axons")
    ax.set_xlabel("X Coordinate [um]")
    ax.set_ylabel("Y Coordinate [um]")
    _apply_axis_limits(ax, config, area_drawn)

    out_path = Path(out_path)
    save(fig, out_path, dpi=DPI, bbox_inches='tight')
    print(f"Axon network visualization saved to {out_path}")

    # 凡例は別ファイル。**名前は本体から導く** (呼び出し側が 2 つ渡さなくて済むように)。
    legend_path = out_path.with_name(f"{out_path.stem}_legend{out_path.suffix}")
    _save_legend_figure(handles, legend_path)
    print(f"Axon network legend saved to {legend_path}")
