"""エリア (ニューロンを配置し軸索を閉じ込める 2D 領域) の可視化。

**符号付き距離関数 (SDF) を格子上で評価して描く。** `BaseArea` の抽象メソッドは `sdf()`
だけなので、形状に依存しない描画方法はこれしかない。`patches.Circle` / `Rectangle` を使うと
円板と矩形は描けても `CompositeArea` (複雑形状を表現する唯一の手段) が描けず本末転倒になる。

    d = area.sdf(格子点)          # 負=内部、正=外部
    contourf(..., levels=[d_min, 0])   # 領域の塗り
    contour (..., levels=[0])          # 境界線

`distributions.py` と同じく「ax を取るプリミティブ + out_path を取る図ラッパ」に分けてある。
プリミティブがあることで `network()` が自分の Axes に境界線を重ねられる。

エリアは**ダックタイピングで受ける** (型注釈を付けない)。必要なのは `sdf` / `bounds` /
`is_bounded`、任意で `part_of` / `part_names` / `area_um2` だけなので、
`src/models/network/area.py` を import して `src/utils/plotting` -> `src/models` という依存を
作る必要がない (`network()` が layout と config を注釈無しで受けているのと同じ作法)。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting.network import _save

# 境界箱をどれだけ広げて格子を張るか (span に対する比)。
# **0 にしてはいけない。** `rect` のように領域が境界箱いっぱいに広がる形状では、
# sdf=0 の等高線がちょうど格子の端に載ってしまい contour が閉じない (実測: pad=0 で
# rect の内部率 100%、境界が格子端に接触)。0.04 で全プロファイルの接触が解消する。
DEFAULT_PAD = 0.04

# 1 辺あたりの格子点数。3 mm 幅の領域で 400 なら刻みは 7.5 um。
DEFAULT_GRID = 400

# part ごとに塗り分けるときの色順 (matplotlib の tab10)。
PART_CMAP = "tab10"


def _padded_bounds(area, pad: float) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.asarray(area.bounds, dtype=np.float64)
    span = hi - lo
    return lo - span * pad, hi + span * pad


def _place_part_labels(ax, labels, span: float, zorder: float) -> None:
    """part 名を重心へ置く。重心が重なるものは縦にずらす。

    交差するブリッジのように**重心が一致する part は普通に存在する** (modular_4 の
    十字は縦棒と横棒の重心がどちらも原点)。素直に重心へ置くと文字が完全に重なって
    片方が読めなくなるので、近すぎるラベルは順に下へ逃がす。
    """
    min_sep = 0.05 * span
    placed: list[tuple[float, float]] = []
    for x, y, name, color in labels:
        while any(np.hypot(x - px, y - py) < min_sep for px, py in placed):
            y -= min_sep
        placed.append((x, y))
        ax.text(x, y, name, ha="center", va="center", fontsize=9, fontweight="bold",
                color=color, zorder=zorder)


def draw_area(
    ax,
    area,
    *,
    grid: int = DEFAULT_GRID,
    pad: float = DEFAULT_PAD,
    fill: bool = True,
    boundary: bool = True,
    by_part: bool | None = None,
    fill_color: str = "tab:blue",
    fill_alpha: float = 0.12,
    part_alpha: float = 0.35,
    edge_color: str = "black",
    linewidth: float = 1.2,
    zorder: float = 0,
    set_limits: bool = True,
) -> bool:
    """エリアの領域と境界線を `ax` へ描く。

    Args:
        ax: 描画先の Axes。図の生成と保存は呼び出し側の責任。
        area: `sdf(points)` / `bounds` / `is_bounded` を持つオブジェクト。
        grid: 1 辺あたりの格子点数。境界線の滑らかさを決める。
        pad: 境界箱を広げる比。0 にすると境界線が閉じないことがある (DEFAULT_PAD 参照)。
        fill: 領域を塗るか。
        boundary: sdf=0 の等高線を引くか。
        by_part: 複合領域を part ごとに色分けするか。None は自動 (`part_of` があれば True)。
        fill_alpha: 単色で塗るときの不透明度。
        part_alpha: part ごとに塗り分けるときの不透明度。
        set_limits: 軸範囲をパディング込みの境界箱に合わせるか。

    Returns:
        描けたら True。無界のエリア (`no_space`) は描きようがないので何もせず False。
    """
    if area is None or not area.is_bounded:
        return False

    lo, hi = _padded_bounds(area, pad)
    xs = np.linspace(lo[0], hi[0], grid)
    ys = np.linspace(lo[1], hi[1], grid)
    X, Y = np.meshgrid(xs, ys)
    points = np.stack([X.ravel(), Y.ravel()], axis=1)

    d = np.asarray(area.sdf(points), dtype=np.float64).reshape(X.shape)
    inside = d <= 0.0
    if not inside.any():
        # 領域が格子の解像度より細い。塗っても線を引いても何も出ないので知らせる。
        raise ValueError(
            f"エリアの内部が格子上に 1 点も現れませんでした (grid={grid})。"
            " grid を上げるか、bounds と sdf の整合を確認してください。"
        )

    part_of = getattr(area, "part_of", None)
    if by_part is None:
        by_part = callable(part_of)

    if fill and by_part and callable(part_of):
        # part 別の塗り分け。part_of() は**領域外の点にも index を返す**ので、
        # contains (= d <= 0) で必ずマスクする。しないと境界箱全体が塗られる。
        idx = np.asarray(part_of(points)).reshape(X.shape)
        names = list(getattr(area, "part_names", []) or [])
        num_parts = int(idx.max()) + 1 if idx.size else 0
        cmap = plt.get_cmap(PART_CMAP)
        labels: list[tuple[float, float, str, tuple]] = []
        for p in range(num_parts):
            mask = inside & (idx == p)
            if not mask.any():
                continue
            # 各 part を「その part に属し、かつ領域内」の指示関数として塗る。
            # 0.5 を境にすることで、mask の縁がそのまま part の切れ目になる。
            ax.contourf(
                X, Y, mask.astype(np.float64), levels=[0.5, 1.5],
                colors=[cmap(p % cmap.N)], alpha=part_alpha, zorder=zorder,
            )
            # ラベルは part の重心へ。part は円や矩形なので重心は内部に入る。
            name = names[p] if p < len(names) else f"M{p}"
            labels.append((float(X[mask].mean()), float(Y[mask].mean()),
                           name, cmap(p % cmap.N)))
        _place_part_labels(ax, labels, span=float(np.max(hi - lo)), zorder=zorder + 3)
    elif fill:
        # 単色の塗り。下端レベルは有限の最小値から取る (有界エリアの sdf に
        # 非有限値は現れないことを確認済み)。
        ax.contourf(
            X, Y, d, levels=[float(d.min()) - 1.0, 0.0],
            colors=[fill_color], alpha=fill_alpha, zorder=zorder,
        )

    if boundary:
        ax.contour(X, Y, d, levels=[0.0], colors=[edge_color],
                   linewidths=linewidth, zorder=zorder + 2)

    ax.set_aspect("equal")
    if set_limits:
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
    return True


def plot_area(area, out_path: Path, *, title: str = "Area", **kwargs) -> bool:
    """エリア**単体**の図を 1 枚保存する。

    ここに描くのは領域だけ。細胞体と結合は `network(..., area=area)` が境界線の上へ
    重ねる担当なので、この図には出さない (「領域そのものの確認」と「ネットワークが
    領域の中でどうなっているかの確認」を別の図に分ける)。

    Args:
        area: 描くエリア。
        out_path: 保存先の png。
        title: 図のタイトル。面積と、`area.num_neurons` が分かれば密度を副題に添える。

    Returns:
        保存したら True。無界のエリアは描けないので何もせず False。
    """
    if area is None or not area.is_bounded:
        return False

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_area(ax, area, **kwargs)

    # 面積と実効密度。密度は soma の座標ではなく area.num_neurons から出す
    # (この図は領域だけを見せるので、座標を受け取らずに済ませる)。
    area_um2 = area.area_um2
    num_neurons = getattr(area, "num_neurons", 0)
    subtitle = f"{area_um2 * 1e-6:.3f} mm$^2$" if area_um2 else "area unknown"
    if area_um2 and num_neurons:
        subtitle += f",  {num_neurons / area_um2 * 1e6:.0f} neurons/mm$^2$"
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("X Coordinate [um]")
    ax.set_ylabel("Y Coordinate [um]")

    _save(fig, Path(out_path))
    return True
