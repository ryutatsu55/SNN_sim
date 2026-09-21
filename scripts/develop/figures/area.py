"""ニューロンが置かれる領域 (area) の形。

`draw_area` は与えられた Axes に境界線を重ねるプリミティブで、`network.py` が細胞体と
結合をその上に描くのに使う。`plot_area` は領域だけの 1 枚。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scripts.develop.figures import save


DEFAULT_PAD = 0.04

# 1 辺あたりの格子点数。3 mm 幅の領域で 400 なら刻みは 7.5 um。
DEFAULT_GRID = 400
# part ごとに塗り分けるときの色順 (matplotlib の tab10)。
PART_CMAP = "tab10"
# soma を置けない part (`allow_soma: false`) の描き方。軸索だけが通れる通路であることが
# 一目で分かるように、塗りを薄くしてハッチを掛ける。
NO_SOMA_HATCH = "///"
NO_SOMA_ALPHA_SCALE = 0.4


# 保存時の解像度。**引数にしない** —— 変えたくなったらここを直す。
DPI = 200
FIGSIZE = (8, 8)


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
        # soma を置けない part は薄く + ハッチ。属性が無いエリアでは全 part が soma 可。
        allows_soma = list(getattr(area, "part_allows_soma", []) or [])
        num_parts = int(idx.max()) + 1 if idx.size else 0
        cmap = plt.get_cmap(PART_CMAP)
        labels: list[tuple[float, float, str, tuple]] = []
        for p in range(num_parts):
            mask = inside & (idx == p)
            if not mask.any():
                continue
            no_soma = p < len(allows_soma) and not allows_soma[p]
            # 各 part を「その part に属し、かつ領域内」の指示関数として塗る。
            # 0.5 を境にすることで、mask の縁がそのまま part の切れ目になる。
            # hatches は list でないと contourf.draw() が落ちるので、soma 不可のときだけ渡す。
            ax.contourf(
                X, Y, mask.astype(np.float64), levels=[0.5, 1.5],
                colors=[cmap(p % cmap.N)], zorder=zorder,
                alpha=part_alpha * NO_SOMA_ALPHA_SCALE if no_soma else part_alpha,
                **({"hatches": [NO_SOMA_HATCH]} if no_soma else {}),
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

def area_figure(built, out_path: Path) -> None:
    """エリア**単体**の図を 1 枚保存する。

    ここに描くのは領域だけ。細胞体と結合は `network()` が境界線の上へ重ねる担当なので、
    この図には出さない (「領域そのものの確認」と「ネットワークが領域の中でどう
    なっているかの確認」を別の図に分ける)。

    無界のエリア (`no_space` など) は `built.area()` が `MissingData` を投げるので、
    ここに分岐は書かない。
    """
    area = built.area()
    profile_name = built.config.network.area.profile_name

    fig, ax = plt.subplots(figsize=FIGSIZE)
    draw_area(ax, area)

    # 面積と実効密度。密度は soma の座標ではなく area.num_neurons から出す
    # (この図は領域だけを見せるので、座標を受け取らずに済ませる)。
    # soma を置けない part があるエリアでは、密度の分母は soma 配置領域の面積になる
    # (ニューロンはそこにしか居ない) ので、幾何全体の面積と両方を出す。
    area_um2 = area.area_um2
    soma_region = getattr(area, "soma_area", area)
    soma_um2 = soma_region.area_um2 if soma_region is not area else area_um2
    num_neurons = getattr(area, "num_neurons", 0)
    subtitle = f"{area_um2 * 1e-6:.3f} mm$^2$" if area_um2 else "area unknown"
    if soma_region is not area and soma_um2:
        subtitle += f"  (soma {soma_um2 * 1e-6:.3f} mm$^2$)"
    if soma_um2 and num_neurons:
        subtitle += f",  {num_neurons / soma_um2 * 1e6:.0f} neurons/mm$^2$"
    ax.set_title(f"Area: {profile_name}\n{subtitle}")
    ax.set_xlabel("X Coordinate [um]")
    ax.set_ylabel("Y Coordinate [um]")

    save(fig, Path(out_path), dpi=DPI)
