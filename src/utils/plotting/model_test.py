"""モデル単体を手で駆動した結果を描く図。**run の図ではない。**

`src/utils/plotting/` の他のモジュールが `(view, out_path)` の契約
(`src/utils/runview.py`) を取るのに対し、ここだけは生の配列を受け取る。理由は
**描く対象が run ではない**から —— 呼ぶのは `scripts/tools/pipeline_check.py` と `test/models/` の
手動テストで、どれも DataLoader を 1 本回しながら毎ステップ `pull()` した値を
メモリに溜めているだけで、run ディレクトリも記録窓も記録ファイル名の規約も持たない。
契約に載せても `run_dir` と `layout` が埋まらず、形だけの view になる。

契約を取る図と混ざらないよう、ファイルを分けてある。**記録を持つ実験の図をここへ
足さないこと** —— それは契約に載る図であって、`traces.py` や `raster.py` の側に属する。
"""

import matplotlib.pyplot as plt
import numpy as np

from src.utils.plotting.common import save_figure


def PQN_test(V_data, I_in, config, out_path):
    tmax = config.task.duration/1000
    time_axis = np.arange(len(V_data)) * config.simulation.dt / 1000.0

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8, 4), sharex=True)

    ax1.plot(time_axis, V_data, color='tab:blue')
    ax1.set_ylabel('v')
    ax1.set_xlim(0, tmax)

    ax2.plot(time_axis, I_in, color='black')
    ax2.set_ylabel('I')
    ax2.set_xlabel('[s]')
    ax2.set_xlim(0, tmax)

    save_figure(fig, out_path, dpi=None)


def neuron_test(V_data, I_in, spike_times, spike_ids, config, out_path, *,
                id=0, x_data=None):
    V_data = V_data[:, id]
    I_in = I_in[:, id]
    tmax = config.task.duration / 1000
    time_axis = np.arange(len(V_data)) * config.simulation.dt / 1000.0

    if x_data is not None:
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(
            4, 1,
            gridspec_kw={'height_ratios': [1, 4, 1, 1]},
            figsize=(9, 5), sharex=True
        )
    else:
        fig, (ax0, ax1, ax2) = plt.subplots(
            3, 1,
            gridspec_kw={'height_ratios': [1, 4, 1]},
            figsize=(9, 4), sharex=True
        )

    spike_times = spike_times / 1000.0
    ax0.scatter(spike_times, spike_ids, s=10)
    ax0.set_ylabel("Neuron ID")
    ax0.set_ylim(id - 0.5, id + 0.5)

    ax1.plot(time_axis, V_data, color='tab:blue')
    ax1.set_ylabel('v [mV]')
    ax1.set_xlim(0, tmax)

    ax2.plot(time_axis, I_in, color='black')
    ax2.set_ylabel('I [nA]')
    ax2.set_xlim(0, tmax)

    if x_data is not None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for neuron_id in range(x_data.shape[1]):
            col = x_data[:, neuron_id]
            if np.all(np.isnan(col)):
                continue
            ax3.plot(time_axis, col, color=colors[neuron_id % len(colors)], label=f'neuron {neuron_id}')
        ax3.set_ylabel('x (STP)')
        ax3.set_xlabel('Time [s]')
        ax3.set_xlim(0, tmax)
        ax3.legend(fontsize=7, loc='upper right')
    else:
        ax2.set_xlabel('Time [s]')

    save_figure(fig, out_path, dpi=None)


def stdp_window(dw: np.ndarray, dt: np.ndarray, out_path, *, title="stdp_window"):
    """
    STDPの学習特性（Δt vs Δw）をプロットし、画像として保存する関数。

    Args:
        dw: 重みの変化量 (Δw = w_after - w_before) の配列
        dt: スパイク時間差 (Δt = t_post - t_pre) [ms] の配列
        out_path: 保存先ファイルパス
        title: グラフのタイトル
    """
    fig = plt.figure(figsize=(10, 6))

    # 0点を強調するガイドライン
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.axvline(0, color='black', linewidth=1, linestyle='--')

    # データの散布図と近似線
    plt.scatter(dt, dw, color='blue', alpha=0.6, label='Measured Data')

    # スムーズな曲線を描くためのソート処理（プロット用）
    sort_idx = np.argsort(dt)
    plt.plot(dt[sort_idx], dw[sort_idx], color='red', linewidth=2, label='STDP Curve')

    # 軸ラベルの設定
    plt.xlabel(r'Spike Timing Difference: $\Delta t$ [ms]', fontsize=12)
    plt.ylabel(r'Weight Change: $\Delta w$ (or $\Delta g$)', fontsize=12)
    plt.title(title, fontsize=14)

    # 領域の解説（LTP/LTD）
    plt.text(max(dt)*0.7, max(dw)*0.1, 'LTP (Potentiation)', fontsize=10, color='green', fontweight='bold')
    plt.text(min(dt)*0.7, max(dw)*0.1, 'LTD (Depression)', fontsize=10, color='orange', fontweight='bold')

    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend()

    # tight_layout は掛けない (元からの体裁を保つため)。
    save_figure(fig, out_path, dpi=300, tight_layout=False)
    print(f"  [Visualization] STDP window plot saved to: {out_path}")
