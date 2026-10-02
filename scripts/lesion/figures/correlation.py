"""ニューロン間相関行列の imshow。記録窓 1 つぶんの 1 枚。

Watanabe et al. (2026) Fig. 2b に合わせた図。相関の定義は
`src/utils/analysis/correlation.py` (発火率時系列のゼロラグ Pearson 相関)。

並べ替えは `style.py` の仕組みに乗り、**重み行列・粗視化結合図と同じ module > polarity**
でブロック化する。論文の A/B/C/D のブラケットに当たるのがこの module ブロックで、
「切ったモジュールの行と列だけが暗くなる」という読み方ができるようにするため。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.analysis.correlation import (RATE_BIN_MS, mean_of_matrix,
                                            neuronal_correlation)
from scripts.lesion.figures import save, style

# 並べ替え軸。**重み行列・粗視化結合図と同じ。** 3 枚が同じブロック配置になるので、
# 「どの群が繋がっているか」「その重みはいくつか」「実際に同期しているか」を
# 位置で突き合わせられる。
ORDER_AXES = style.ORDER_AXES

# 見た目。**論文 Fig. 2b と同じ viridis の [0, 1]。** 図を並べて比べるための固定値。
#
# **負の相関は色の下端に潰れる** (論文の色範囲に合わせた結果)。負のペアは実測で
# 全体の 1 割前後あるので、情報を捨てている。**これを認めているのはこの図だけ** ——
# `neuronal_correlation()` が返す値も `metrics.csv` の `mean_correlation` も符号付きのまま
# で、相関を描く図を新しく足すときにこの約束を引き継がないこと。
SINGLE_FIGSIZE = (6, 5.4)
VMIN, VMAX = 0.0, 1.0
CMAP = "viridis"
# 未定義 (窓の中で 1 発も撃たないニューロン) だけは色を分ける。**相関 0 とは別物**で、
# 「切った先が黙った」のか「切った先が非同期になった」のかがこの色で分かれる。
UNDEFINED_COLOR = "0.85"
DPI = 200

# N² の行列を画像にする上限。重み行列と同じ理由・同じ値。
DENSE_RENDER_LIMIT = 20000
# 最外ブロックの名前を目盛りに出す上限。これを超えると文字が潰れるので番号のままにする。
MAX_BLOCK_TICKS = 16


def correlation_matrix(window, out_path: Path) -> None:
    """1 記録窓ぶんのニューロン間相関行列を可視化する。"""
    layout = window.layout
    total = int(window.total_neurons)
    if total > DENSE_RENDER_LIMIT:
        raise MemoryError(
            f"N={total} の相関行列を画像にするには密な "
            f"{total**2 * 8 / 2**30:.1f} GiB が必要です (上限 N={DENSE_RENDER_LIMIT})。"
        )

    spikes = window.spikes()
    matrix = neuronal_correlation(spikes.times, spikes.ids, total,
                                  window.record_window_ms)
    ordering = style.resolve_ordering(layout, style.available_order_axes(layout, ORDER_AXES))
    ordered = ordering.apply(matrix)

    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(UNDEFINED_COLOR)

    fig, ax = plt.subplots(figsize=SINGLE_FIGSIZE)
    image = ax.imshow(ordered, origin="upper", interpolation="nearest",
                      vmin=VMIN, vmax=VMAX, cmap=cmap)

    if ordering.enabled:
        # 線は**最外ブロック (module) の境界だけ**。重み行列と同じ約束。
        style.draw_block_boundaries(ax, ordering, ordered.shape[0], skip=("polarity",))
        blocks = style.outer_block_labels(layout, ordering, ordered.shape[0])
        if blocks and len(blocks) <= MAX_BLOCK_TICKS:
            ticks = [centre - 0.5 for centre, _ in blocks]
            names = [name for _, name in blocks]
            ax.set_xticks(ticks)
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_yticks(ticks)
            ax.set_yticklabels(names)
        else:
            ticks = style.block_ticks(ordering, ordered.shape[0])
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

    # 平均相関はこの図の要約そのもの。metrics.csv を開かずに 2 枚を比べられるよう
    # タイトルに出す (値の正は metrics.csv の mean_correlation)。
    ax.set_title(f"Correlation {window.label}\n"
                 f"mean = {mean_of_matrix(ordered):.3f}  ({RATE_BIN_MS:g} ms bins)")
    ax.set_xlabel("neuron id")
    ax.set_ylabel("neuron id")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="correlation")
    save(fig, out_path, dpi=DPI)
