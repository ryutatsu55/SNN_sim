"""単一ニューロンの膜電位トレース。

**run の図。** 記録窓 1 つぶんの `Trace` (V / Isyn / その窓のスパイク) を描く。
x 軸の上限を `config.task.duration` ではなくトレース窓の幅で固定するため、長時間
シミュレーション (72h 等) から切り出した短い窓をそのまま表示できる。

モデル単体を手で駆動した結果を描く図 (`neuron_test` / `PQN_test` / `stdp_window`) は
`model_test.py` にある。あちらは run ではないので契約を取らない。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting.common import save_figure


def neuron_trace(window, out_path: Path) -> None:
    """単一ニューロンの膜電位・シナプス電流・スパイクを固定時間窓で描く。

    レイアウトは `model_test.neuron_test` と同じ (ラスター / V / I) で、x 軸の上限が
    `config.task.duration` ではなくトレース窓の幅。
    """
    trace = window.trace()
    V = np.asarray(trace.V)
    I_in = trace.I
    time_axis = np.arange(len(V)) * trace.dt / 1000.0

    if I_in is not None:
        fig, (ax0, ax1, ax2) = plt.subplots(
            3, 1, gridspec_kw={"height_ratios": [1, 4, 1]},
            figsize=(9, 4), sharex=True)
    else:
        fig, (ax0, ax1) = plt.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 4]},
            figsize=(9, 3.5), sharex=True)

    ax0.scatter(np.asarray(trace.spike_times) / 1000.0, trace.spike_ids, s=10)
    ax0.set_ylabel("Neuron ID")

    ax1.plot(time_axis, V, color="tab:blue")
    ax1.set_ylabel("v [mV]")
    ax1.set_xlim(0, trace.window_s)

    if I_in is not None:
        ax2.plot(time_axis, np.asarray(I_in), color="black")
        ax2.set_ylabel("I [nA]")
        ax2.set_xlim(0, trace.window_s)
        ax2.set_xlabel("Time [s]")
    else:
        ax1.set_xlabel("Time [s]")

    save_figure(fig, Path(out_path), dpi=None)
