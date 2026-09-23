"""シナプス量 (重み・遅延・距離) のヒストグラム。

全体に加えて E/I ブロック別のパネルを添える。骨格は `_value_distribution` 1 つで、
重みも遅延も距離もその薄い包み。入力は COO (row, col と index 整合の 1D 配列) なので、
実在する結合の値だけが数えられる —— 結合の無い箇所の 0 が分布に山を作ることはない。

**時点を持つのは重みだけ。** 遅延と距離は run を通して不変なので構造図が 1 枚出すが、
重みは build 直後 (構造図) と記録時刻ごと (パネル) の両方から同じ関数が呼ばれる。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.utils.analysis.weights import (
    BLOCK_ORDER,
    block_masks,
    excitatory_flags,
    synapse_distances,
)
from src.utils.runview import MissingData
from scripts.develop.figures import save
from scripts.develop.figures.style import BLOCK_COLORS


# 保存時の解像度。**引数にしない** —— 変えたくなったらここを直す。
DPI = 200
BINS = 80
HIST_FIGSIZE = (11, 4)
WEIGHT_TITLE = "Weight distribution"


def _value_distribution(view, values: np.ndarray, out_path: Path, *,
                        xlabel: str, title: str, unit: str = "") -> None:
    """COO 上の per-synapse 量のヒストグラム (左: 全体 / 右: E/I ブロック別)。

    遅延・距離・重みの共通の骨格。右パネルは左と**同じビン境界**を使うので、
    ブロック別の山が全体のどこに乗っているか読める。

    Args:
        view: 契約の `Built` か `Window`。row/col と E/I の分類をここから取る。
        values: 各シナプスの値 (1D, wiring と index 整合)
        xlabel: 横軸ラベル (単位を含めて呼び出し側が決める)
        title: 図全体のタイトル
        unit: 左パネルの mean/max に添える単位。空なら数値だけ。
    """
    wiring = view.wiring()
    if wiring.num_synapses == 0:
        raise MissingData("synapses", "結合が 1 本もありません")
    values = np.asarray(values, dtype=np.float64)
    masks = block_masks(wiring.row, wiring.col,
                        excitatory_flags(view.layout, view.total_neurons))
    suffix = f" {unit}" if unit else ""

    fig, axes = plt.subplots(1, 2, figsize=HIST_FIGSIZE)

    axes[0].hist(values, bins=BINS, color="black")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Number of synapses")
    if values.size:
        axes[0].set_title(f"All synapses (n={values.size})\n"
                          f"mean={values.mean():.2f}{suffix}, max={values.max():.2f}{suffix}")
    else:
        axes[0].set_title("All synapses (empty)")

    edges = np.histogram_bin_edges(values, bins=BINS) if values.size else np.linspace(0, 1, BINS)
    drawn = 0
    for name in BLOCK_ORDER:
        block = values[masks[name]]
        if block.size:
            axes[1].hist(block, bins=edges, histtype="step", lw=1.4,
                         color=BLOCK_COLORS[name], label=f"{name} (n={block.size})")
            drawn += 1
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Number of synapses")
    axes[1].set_title("By connection type")
    if drawn:
        # 1 本も描いていないときに legend() を呼ぶと matplotlib が警告を出すだけなので黙る。
        axes[1].legend(fontsize=7)

    fig.suptitle(title)
    save(fig, out_path, dpi=DPI)

def delay_distribution(built, out_path: Path) -> None:
    """実在する結合上の伝播遅延のヒストグラム (全体 + E/I ブロック別)。

    結合が無い箇所は行列上 0 で埋まるため、必ず COO (= 実結合のみ) を渡すこと。
    """
    _value_distribution(
        built, built.coo().delays, out_path,
        xlabel="Delay [ms]", title="Delay distribution", unit="ms",
    )

def distance_distribution(built, out_path: Path) -> None:
    """実在する結合の**長さ**のヒストグラム (全体 + E/I ブロック別)。

    `delay: distance_based` なら遅延の図と相似形になる。ずれたときは、遅延だけ頭打ち
    なら `max_delay` の clip、距離だけ広がっているなら伝導速度の設定。

    `empirical_connection_probability` とは分母が違う。あちらは確率
    (その距離のペアのうち何割が繋がったか)、こちらは件数。
    """
    wiring = built.wiring()
    _value_distribution(
        built, synapse_distances(built.coords(), wiring.row, wiring.col), out_path,
        xlabel="Distance [um]", title="Synapse distance distribution", unit="um",
    )

def _at(view) -> str:
    """その view が指す時点。`Built` は時点を持たないので空文字になる。"""
    hour = getattr(view, "hour", None)
    return "" if hour is None else f" ({hour:g} h)"

def weight_distribution(view, out_path: Path) -> None:
    """実在する結合上の重みのヒストグラム (全体 + E/I ブロック別)。

    **`Built` にも `Window` にも渡せる。** 構造図は build 直後の初期重み、パネルは
    その記録時刻の重みを描く。どちらの時点かはタイトルに入る。

    時間をまたいだ比較は `fig2c.py` の軌跡と `weight_matrix.py` の行列が受け持つ。
    """
    _value_distribution(
        view, view.coo().weights, out_path,
        xlabel="Weight", title=f"{WEIGHT_TITLE}{_at(view)}",
    )
