"""単一ニューロンの膜電位・シナプス電流・スパイクを固定時間窓で描く。

x 軸の上限を `config.task.duration` ではなく引数 `window_s` で固定するため、長時間
シミュレーション (72h 等) から切り出した短い窓をそのまま表示できる。
"""
import matplotlib.pyplot as plt
import numpy as np
from scripts.develop.figures import save


def neuron_trace(window, out_path):
    """単一ニューロンの膜電位・シナプス電流・スパイクを固定時間窓で描画する。

    neuron_test と同じレイアウト (ラスター / V / I) だが、x 軸上限を
    `config.task.duration` ではなくトレース窓の幅で固定するため、長時間
    シミュレーション (72h 等) から切り出した短い窓をそのまま表示できる。

    トレースを採っていない run では `window.trace()` が `MissingData` を投げ、
    登録簿がそれを受けてこの図を飛ばす。**ここに分岐は書かない。**
    """
    trace = window.trace()
    V, I_in = trace.V, trace.I
    spike_times, spike_ids = trace.spike_times, trace.spike_ids
    dt, neuron_id, window_s = trace.dt, trace.neuron_id, trace.window_s

    V = np.asarray(V)
    time_axis = np.arange(len(V)) * dt / 1000.0

    if I_in is not None:
        fig, (ax0, ax1, ax2) = plt.subplots(
            3, 1,
            gridspec_kw={'height_ratios': [1, 4, 1]},
            figsize=(9, 4), sharex=True
        )
    else:
        fig, (ax0, ax1) = plt.subplots(
            2, 1,
            gridspec_kw={'height_ratios': [1, 4]},
            figsize=(9, 3.5), sharex=True
        )

    spike_times = np.asarray(spike_times) / 1000.0
    ax0.scatter(spike_times, spike_ids, s=10)
    ax0.set_ylabel("Neuron ID")

    ax1.plot(time_axis, V, color='tab:blue')
    ax1.set_ylabel('v [mV]')
    ax1.set_xlim(0, window_s)

    if I_in is not None:
        ax2.plot(time_axis, np.asarray(I_in), color='black')
        ax2.set_ylabel('I [nA]')
        ax2.set_xlim(0, window_s)
        ax2.set_xlabel('Time [s]')
    else:
        ax1.set_xlabel('Time [s]')

    save(fig, out_path)
    plt.close()
