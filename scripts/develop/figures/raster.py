"""ラスター図。スパイクを (時刻, ニューロン) の点として描く。

y 軸はグローバル ID 順ではなく `style.available_order_axes()` の軸で並べ替え、
ブロック境界に線を入れる。「モジュールでブロック化し、その中で E→I」のような読み方が
できるようにするため。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.utils.analysis.weights import excitatory_flags
from scripts.develop.figures import save, style

# 論文 (Ikeda-Akita-Takahashi 2023) Fig.2 の時間軸。図を並べて比べるための固定値。
PAPER_XLIM_S = (0.0, 30.0)
MARKER_SIZE = 2.0
FIGSIZE = (10, 4)
DPI = 200


def raster(window, out_path: Path) -> None:
    """記録窓 1 つのラスター図を描く。

    y 軸は `style.available_order_axes()` の順に並べ替え (表示ID は 1 始まり)、
    興奮性=赤 / 抑制性=青で色分けし、**E/I 以外の**ブロックの境目に破線を引く。
    並べ替え軸が 1 つも使えない layout では生のグローバルIDをそのまま y 軸に使う。
    """
    spikes = window.spikes()
    times, ids = spikes.times, spikes.ids
    layout = window.layout
    total_neurons = window.total_neurons
    hour = window.hour

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ordering = style.resolve_ordering(layout, style.available_order_axes(layout))
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

        ax.scatter(t_s[exc_mask], display[exc_mask], s=MARKER_SIZE, color="tab:red", label="Excitatory")
        ax.scatter(t_s[~exc_mask], display[~exc_mask], s=MARKER_SIZE, color="tab:blue", label="Inhibitory")
        for position, level in ordering.visible_boundaries():
            if 0 < position < total:
                ax.axhline(position + 0.5, color="gray",
                           lw=0.8 if level == 0 else 0.5, ls="--")
        if n_exc > 0 and n_inh > 0:
            ax.legend(loc="upper right", markerscale=3, fontsize=8)
        ax.set_ylim(0.5, total + 0.5)
        remapped = True

    if not remapped:
        ax.scatter(times / 1000.0, ids, s=MARKER_SIZE, color="black")
        # 論文の (0, 100) は N=100 のこと。系のサイズに追従させる。
        ax.set_ylim(0.0, float(total_neurons))

    ax.set_title(f"Raster {hour:g} h")
    ax.set_xlabel("Time [s]")
    if not remapped:
        ax.set_ylabel("Neuron ID")
    elif ordering.axes == ("polarity",):
        ax.set_ylabel("Neuron ID (excitatory 1..Nexc, inhibitory above)")
    else:
        ax.set_ylabel(f"Neuron ID (ordered by {' > '.join(ordering.axes)})")
    ax.set_xlim(*PAPER_XLIM_S)
    save(fig, out_path, dpi=DPI)
