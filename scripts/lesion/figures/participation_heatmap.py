"""群ごとの参加率の分布を、切断をまたいだ時刻に沿って並べたヒートマップ。run 全体で 1 枚。

各 probe の窓を `EVENT_BIN_MS` のビンに切り、群の何割がそのビンで 1 発以上撃ったか
(参加率) を数える。**群の誰かが撃ったビンはすべて**数え、同期イベントの閾値
(`ACTIVE_FRACTION`) は線で重ねるだけにする —— 閾値より下も見えるので、「同期が浅く
なった」(山が下へずれる) と「同期が消えた」(上の山が消える) が分布の形で分かれる。

    横軸 = probe の時刻。各列は「その probe から次の probe まで」(切断 = 0)。
           ベースラインは切断の直前に 1 列
    縦軸 = 1 ビンの参加率
    色   = そのビンが 1 秒あたり何回現れたか (対数)

定義は `src/utils/analysis/synchrony.py` の `active_fraction_series`。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from src.utils.analysis.synchrony import (ACTIVE_FRACTION, EVENT_BIN_MS,
                                          active_fraction_series)
from scripts.lesion.figures import save
from scripts.lesion.figures.style import synchrony_groups

# 縦軸のビン数。(0, 1] を等分する。0 (誰も撃たない) は数えない。
NUM_FRACTION_BINS = 20
PANEL_HEIGHT = 2.0
FIGWIDTH = 10
DPI = 200
CMAP = "magma"
CUT_COLOR = "tab:cyan"
THRESHOLD_COLOR = "white"


def _time_edges(hours: np.ndarray) -> np.ndarray:
    """各 probe のセルを「その時刻から次の probe まで」とする境界。

    ベースライン (負の時刻) の列は**切断 (0) の直前に、post の典型的な間隔と同じ幅で**
    置く。実際の時刻 (窓幅ぶん手前) に置くと、切断の線に潰れて読めなくなる。
    """
    post = hours[hours >= 0.0]
    step = float(np.median(np.diff(post))) if post.size > 1 else 1.0
    num_baseline = int(np.count_nonzero(hours < 0.0))
    left = -step * np.arange(num_baseline, 0, -1)
    return np.concatenate([left, post, [post[-1] + step] if post.size else [0.0]])


def participation_heatmap(series, out_path: Path) -> None:
    """全体と群ごとに 1 段ずつ、参加率の分布の時間変化を描く。"""
    windows = sorted(series.windows, key=lambda window: window.hour)
    groups = synchrony_groups(series.layout)
    hours = np.array([window.hour for window in windows], dtype=np.float64)
    fraction_edges = np.linspace(0.0, 1.0, NUM_FRACTION_BINS + 1)

    # rates[group][fraction_bin, probe] = 1 秒あたりの出現回数
    rates = {name: np.zeros((NUM_FRACTION_BINS, len(windows))) for name in groups}
    for column, window in enumerate(windows):
        spikes = window.spikes()
        seconds = window.record_window_ms / 1000.0
        for name, members in groups.items():
            fraction = active_fraction_series(spikes.times, spikes.ids, members,
                                              window.record_window_ms)
            counts, _ = np.histogram(fraction[fraction > 0], bins=fraction_edges)
            rates[name][:, column] = counts / seconds

    positive = np.concatenate([values[values > 0] for values in rates.values()])
    norm = (LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
            if positive.size else None)

    fig, axes = plt.subplots(len(groups), 1, sharex=True,
                             figsize=(FIGWIDTH, PANEL_HEIGHT * len(groups)), squeeze=False)
    time_edges = _time_edges(hours)
    image = None
    for ax, (name, members) in zip(axes[:, 0], groups.items()):
        # 0 回のセルは塗らない (対数の色に 0 は無い)。背景色で「1 度も無い」と読める。
        image = ax.pcolormesh(time_edges, fraction_edges,
                              np.ma.masked_less_equal(rates[name], 0.0),
                              cmap=CMAP, norm=norm, shading="flat")
        ax.set_facecolor("0.15")
        ax.axhline(ACTIVE_FRACTION, color=THRESHOLD_COLOR, linestyle=":", linewidth=1.0)
        ax.axvline(0.0, color=CUT_COLOR, linestyle="--", linewidth=1.2)
        ax.set_ylabel(f"{name}\n(n={members.size})")
        ax.set_ylim(0.0, 1.0)
    axes[-1, 0].set_xlabel("Time since lesion [h]")
    axes[0, 0].set_title(
        f"Participation per {EVENT_BIN_MS:g} ms bin (fraction of the group that fired)\n"
        f"dotted = event threshold {ACTIVE_FRACTION:g}, dashed = lesion,"
        " column left of the lesion = baseline from parent run", fontsize=10)
    if image is not None:
        fig.colorbar(image, ax=axes[:, 0].tolist(), fraction=0.03, pad=0.02,
                     label="bins per second")
    save(fig, out_path, tight_layout=False, dpi=DPI, bbox_inches="tight")
