"""論文 (Ikeda-Akita-Takahashi 2023) Fig.2(c) 相当の図。

左列にシナプス重みの発達 (E/I ブロック別の束)、右列にネットワーク指標の推移を並べる。

重みの軌跡を組み立てる `_weight_trajectories()` もここにある。**run 全体を通した計算は、
それを使う図が持つ** —— 読み手 (`store/`) に加工を混ぜない。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.analysis.powerlaw import llr_ceiling_per_avalanche
from src.utils.analysis.weights import BLOCK_ORDER, block_masks, excitatory_flags
from scripts.akita_soc.figures import save

# 左列。`_weight_trajectories()` が返す鍵の順。
WEIGHT_BLOCKS = ("W_EE", "W_EI", "W_IE", "W_II")
# 右列。(metrics.csv の列名, 縦軸ラベル)
METRIC_ROWS = (
    ("llr", "LLR"),
    ("bimodality_d", "D"),
    ("delta_cr", "ΔCr"),
    ("burstiness_index", "BI"),
)
# 線が濃すぎる / 薄すぎるときに触る。シナプス数で見え方が変わる。
TRAJECTORY_ALPHA = 0.02
FIGSIZE = (12, 10)
DPI = 300


def fig2c(series, out_path: Path) -> None:
    """`series` の重み軌跡と指標を 4x2 のパネルに描く。"""
    df = series.metrics()
    hours, trajectories = _weight_trajectories(series)

    if len(df) != len(hours):
        # 指標と npz が同じ記録時刻を指していない。片方だけ作り直した run。
        raise ValueError(
            f"metrics.csv の行数 {len(df)} が記録時刻の数 {len(hours)} と一致しません。"
            " 再解析 (python -m scripts.akita_soc.replot) で作り直してください。"
        )

    fig, axes = plt.subplots(4, 2, figsize=FIGSIZE, sharex='col')
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    # --- 左列: シナプス重みの時間発展 (1 本 1 本を薄い線で重ねる) ---
    for i, block in enumerate(WEIGHT_BLOCKS):
        ax = axes[i, 0]
        # X(1 次元) と Y(2 次元) を渡すと Y の列数ぶんの線が引かれる。
        ax.plot(hours, trajectories[block], color='black',
                alpha=TRAJECTORY_ALPHA, linewidth=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_ylabel(block)
        if i == 0:
            ax.set_title('Synaptic Weights Development')

    # --- 右列: ネットワーク指標の推移 ---
    for i, (column, ylabel) in enumerate(METRIC_ROWS):
        ax = axes[i, 1]
        if column not in df.columns:
            continue
        ax.plot(hours, df[column], color='black', linewidth=1.5)

        if column == 'llr':
            _draw_llr_ceiling(ax, hours, df)
        elif column == 'delta_cr':
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        elif column == 'burstiness_index':
            top = max(df[column]) if max(df[column]) > 0 else 0.5
            ax.set_ylim(0, top * 1.2)

        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.set_title('Network Characteristics')

    for ax in (axes[3, 0], axes[3, 1]):
        ax.set_xlabel('Time (h)')
        # 記録時刻が 1 点だけの run では幅 0 になるので matplotlib に任せる。
        if hours.size and hours.max() > 0:
            ax.set_xlim(0, hours.max())

    save(fig, out_path, tight_layout=False, dpi=DPI, bbox_inches='tight')


def _weight_trajectories(series) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """シナプス 1 本ずつの重みの軌跡を、E/I ブロック別に `[時刻数, シナプス数]` で返す。

    描くのは**実在するシナプス**だけ (COO は実結合しか持たない)。結合の無いペアが
    0 のまま平らな線として束に混ざることはない。
    """
    wiring = series.wiring()
    masks = block_masks(wiring.row, wiring.col,
                        excitatory_flags(series.layout, series.total_neurons))

    stacks: dict[str, list[np.ndarray]] = {name: [] for name in BLOCK_ORDER}
    for window in series.windows:
        values = window.weights()
        for name in BLOCK_ORDER:
            stacks[name].append(values[masks[name]])
    return series.hours, {f"W_{name}": np.array(stacks[name]) for name in BLOCK_ORDER}


def _draw_llr_ceiling(ax, hours, df) -> None:
    """べき乗分布が [1, smax] で出せる LLR の上限を重ねる。

    論文 Fig.2(c) の絶対値はこの線を超えており (72h で 1.65 倍)、**LLR の絶対値は論文と
    比較できない**。自分の値が天井の何割かで読むこと。詳細は `powerlaw.py` の docstring。

    `metrics.csv` の `llr` も同じ [1, smax] で評価してあるので、線と縦軸は同じ土俵。
    """
    if 'avalanche_smax' not in df.columns or 'num_avalanches' not in df.columns:
        return
    smax = int(df['avalanche_smax'].iloc[0])
    ax.plot(hours, llr_ceiling_per_avalanche(smax=smax) * df['num_avalanches'],
            color='tab:green', linestyle='-.', linewidth=1,
            label=f'power-law ceiling (smax={smax})')
    ax.legend(fontsize=7, loc='lower right')
