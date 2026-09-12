"""ラスター図。

`layout` を渡すと y 軸を `order_axes` の順に並べ替え、E/I で色分けして
**E/I 以外の**ブロック境界に線を引く。並べ替えの指定は重み行列 (`plotting.matrices`) と
共通なので、`order_axes=("layer", "polarity")` のような指定が両方の図に同じ意味で効く。

E/I の切り替わりに線を引かないのは、赤/青の色分けが既にその位置を示しているから。
`order_axes=("module", "polarity")` なら線はモジュール境界だけになり、各モジュール帯の
中は色だけで E→I が読める。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.analysis.weights import excitatory_flags
from src.utils.plotting.ordering import DEFAULT_ORDER_AXES, resolve_ordering


def plot_raster(
    times: np.ndarray,
    ids: np.ndarray,
    out_path: Path,
    title: str,
    xlim_s: tuple[float, float] | None = None,
    ylim_neuron: tuple[float, float] | None = None,
    layout=None,
    order_axes: tuple[str, ...] | None = DEFAULT_ORDER_AXES,
    marker_size: float = 2.0,
) -> None:
    """スパイク列のラスター図を描く。

    `layout` (NetworkLayout) を渡すと y 軸を `order_axes` の順に並べ替え (表示ID は
    1 始まり)、興奮性=赤 / 抑制性=青で色分けし、**E/I 以外の**ブロックの境目に破線を
    引く。省略時、または `order_axes=None` のときは生のグローバルIDをそのまま y 軸に使う。

    Args:
        times: スパイク時刻 [ms]
        ids: スパイクを出したニューロンのグローバルID
        out_path: 出力ファイルパス
        title: グラフタイトル
        xlim_s: 時間軸の範囲 [s]
        ylim_neuron: y 軸の範囲。並べ替えを行う場合は無視される (全体を表示する)
        layout: NetworkLayout
        order_axes: 並べ替えに使う軸を外側から順に
        marker_size: 点の大きさ
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))

    ordering = resolve_ordering(layout, order_axes)
    remapped = False
    if ordering.enabled and ordering.rank.size:
        # rank は「グローバルID -> 表示位置(0始まり)」。表示IDは 1 始まりにする。
        display_of = ordering.rank + 1
        total = display_of.size
        gid = np.asarray(ids, dtype=np.int64)
        valid = (gid >= 0) & (gid < total)
        gid = gid[valid]
        t_s = times[valid] / 1000.0
        display = display_of[gid]

        # 色は表示位置ではなく極性そのもので決める。並べ替えが層優先の場合、
        # 興奮性は先頭ブロックに固まらないため。
        is_exc = excitatory_flags(layout, total)
        exc_mask = is_exc[gid]
        n_exc = int(np.count_nonzero(is_exc))
        n_inh = total - n_exc

        ax.scatter(t_s[exc_mask], display[exc_mask], s=marker_size, color="tab:red", label="Excitatory")
        ax.scatter(t_s[~exc_mask], display[~exc_mask], s=marker_size, color="tab:blue", label="Inhibitory")
        for position, level in ordering.visible_boundaries():
            if 0 < position < total:
                ax.axhline(position + 0.5, color="gray",
                           lw=0.8 if level == 0 else 0.5, ls="--")
        if n_exc > 0 and n_inh > 0:
            ax.legend(loc="upper right", markerscale=3, fontsize=8)
        ax.set_ylim(0.5, total + 0.5)
        remapped = True

    if not remapped:
        ax.scatter(times / 1000.0, ids, s=marker_size, color="black")
        if ylim_neuron is not None:
            ax.set_ylim(*ylim_neuron)

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    if not remapped:
        ax.set_ylabel("Neuron ID")
    elif ordering.axes == ("polarity",):
        ax.set_ylabel("Neuron ID (excitatory 1..Nexc, inhibitory above)")
    else:
        ax.set_ylabel(f"Neuron ID (ordered by {' > '.join(ordering.axes)})")
    if xlim_s is not None:
        ax.set_xlim(*xlim_s)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
