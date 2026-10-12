"""単一ニューロンの膜電位トレースを、同じ時間のラスターと並べて描く。

記録窓の先頭で毎ステップ採った全ニューロンの膜電位・入力電流 (`window.voltage()` /
`window.current()`) から、`task.trace_neuron` の行を取り出して描く。x 軸は採った長さで切るので、長時間
シミュレーション (72h 等) の記録窓の先頭だけを表示できる。
"""
import matplotlib.pyplot as plt
import numpy as np
from scripts.develop.figures import save


def neuron_trace(window, out_path):
    """`task.trace_neuron` の膜電位・入力電流を、同じ時間の全体のラスターと並べて描く。

    膜電位を採っていない run では `window.voltage()` が `MissingData` を投げ、
    登録簿がそれを受けてこの図を飛ばす。**ここに分岐は書かない。**
    """
    voltage = window.voltage()
    current = window.current()
    neuron_id = int(window.config.task.trace_neuron)
    duration_ms = voltage.duration_ms
    time_axis = np.arange(voltage.values.shape[1]) * voltage.dt / 1000.0

    spikes = window.spikes()
    head = spikes.times < duration_ms

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1,
        gridspec_kw={'height_ratios': [1, 4, 1]},
        figsize=(9, 4), sharex=True
    )

    ax0.scatter(spikes.times[head] / 1000.0, spikes.ids[head], s=10)
    ax0.set_ylabel("Neuron ID")
    ax0.set_title(f"neuron {neuron_id}")

    ax1.plot(time_axis, voltage.values[neuron_id], color='tab:blue')
    ax1.set_ylabel('v [mV]')

    ax2.plot(time_axis, current.values[neuron_id], color='black')
    ax2.set_ylabel('I [nA]')
    ax2.set_xlim(0, duration_ms / 1000.0)
    ax2.set_xlabel('Time [s]')

    save(fig, out_path)
    plt.close()
