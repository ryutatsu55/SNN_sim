"""切断の瞬間をまたいだ重みの動き。2 枚。

- `weight_trajectories` : シナプス 1 本ずつの軌跡を薄い線で束ねる (develop の fig2c 左列)
- `weight_distribution_shift` : 切断直前 / 直後 / 最終の**分布**を重ねる

**1 ファイルに 2 枚あるのは、どちらも同じ前処理を必要とするから。** 損傷実験には
develop に無い事情が 2 つあり、`_surviving_trajectories()` がその両方を吸収している。

1. **baseline と post でシナプス本数が違う。** baseline は Phase 1 (切断前・全シナプス)、
   post は Phase 2 (切断後)。同じ図に載せるには、生き残ったシナプスへ引き当てる必要が
   ある (`src/utils/analysis/weights.align_subset_to_coo`)。位置で対応づけてはいけない。
2. **時間軸が切断からの相対時間で、負の領域を持つ。** baseline は切断より前の測定。

片方だけ直すと 2 枚が食い違うので、同じファイルに置いてある。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.analysis.weights import BLOCK_ORDER, block_masks, excitatory_flags
from src.utils.analysis.weights import align_subset_to_coo
from scripts.lesion.figures import save

# develop の fig2c と同じ。1 本ずつの線は薄く重ねて「束」として読ませる。
LINE_ALPHA = 0.02
LINE_WIDTH = 0.5
TRAJECTORY_FIGSIZE = (7, 10)
SHIFT_FIGSIZE = (10, 7)
HIST_BINS = 60
DPI = 200
# 重みの上限。可塑性の Wmax で正規化済みなので 1.0。
WMAX = 1.0


def _surviving_trajectories(series) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """**生き残ったシナプス**の重み軌跡を E/I ブロック別に集める。

    Returns:
        (hours, {"W_EE": (T, n_EE), ...})  hours は切断からの経過 [h] (baseline は負)
    """
    post = series.wiring()
    pre = series.wiring_pre()
    masks = block_masks(post.row, post.col,
                        excitatory_flags(series.layout, series.total_neurons))

    hours, stacks = [], {name: [] for name in BLOCK_ORDER}
    for window in series.windows:
        values = window.weights()
        if window.is_baseline:
            # Phase 1 の全シナプスぶんある。生き残った分へ (pre, post) で引き当てる。
            values = align_subset_to_coo(pre.row, pre.col, values,
                                         post.row, post.col, int(post.shape[1]))
        hours.append(window.hour)
        for name in BLOCK_ORDER:
            stacks[name].append(values[masks[name]])
    return (np.asarray(hours, dtype=np.float64),
            {f"W_{name}": np.asarray(stacks[name]) for name in BLOCK_ORDER})


def weight_trajectories(series, out_path: Path) -> None:
    """重み軌跡を E/I ブロック別に 4 段で描く。

    薄い線の束だけだと probe 数が少ないときに読めないので、**ブロック平均を重ねる**。
    切断の瞬間 (t=0) には縦の破線を引く。
    """
    hours, trajectories = _surviving_trajectories(series)

    fig, axes = plt.subplots(4, 1, figsize=TRAJECTORY_FIGSIZE, sharex=True)
    for index, name in enumerate(BLOCK_ORDER):
        ax = axes[index]
        values = trajectories[f"W_{name}"]
        if values.size:
            ax.plot(hours, values, color="black", alpha=LINE_ALPHA, linewidth=LINE_WIDTH)
            ax.plot(hours, values.mean(axis=1), color="tab:red", linewidth=1.6,
                    label=f"mean (n={values.shape[1]})")
            ax.legend(fontsize=7, loc="upper right")
        ax.axvline(0.0, color="tab:blue", linestyle="--", linewidth=1.2)
        ax.set_ylim(-0.05 * WMAX, 1.05 * WMAX)
        ax.set_yticks([0.0, 0.5 * WMAX, WMAX])
        ax.set_ylabel(f"W_{name}")
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_title("Synaptic weights across the lesion\n"
                      "(surviving synapses only; dashed line = lesion)")
    axes[-1].set_xlabel("Time since lesion [h]")
    save(fig, out_path, dpi=DPI)


def weight_distribution_shift(series, out_path: Path) -> None:
    """切断直前 / 直後 / 最終の重み**分布**を 1 枚に重ねる。

    軌跡図が「各シナプスがどう動いたか」を見せるのに対し、こちらは「集団としての形が
    どう変わったか」を見せる。回復が分布の平行移動なのか、両端への二極化なのかは
    軌跡の束からは読み取りにくい。
    """
    hours, trajectories = _surviving_trajectories(series)

    # ラベルは ASCII にする。既定フォント (DejaVu Sans) に CJK が無く、豆腐になるため。
    picks = [(0, "before lesion")]
    if hours.size > 1:
        picks.append((1, "just after"))
    if hours.size > 2:
        picks.append((hours.size - 1, "final"))

    fig, axes = plt.subplots(2, 2, figsize=SHIFT_FIGSIZE)
    edges = np.linspace(0.0, WMAX, HIST_BINS + 1)
    for index, name in enumerate(BLOCK_ORDER):
        ax = axes.flat[index]
        values = trajectories[f"W_{name}"]
        if not values.size:
            ax.set_title(f"W_{name} (0 synapses)")
            continue
        for pick, label in picks:
            ax.hist(values[pick], bins=edges, histtype="step", linewidth=1.4,
                    label=f"{label} ({hours[pick]:+.2f} h)")
        ax.set_title(f"W_{name}  (n={values.shape[1]})")
        ax.set_xlabel("weight")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("Weight distribution before / after the lesion (surviving synapses)")
    save(fig, out_path, dpi=DPI)
