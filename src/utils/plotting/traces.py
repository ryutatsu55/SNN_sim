"""単一ニューロンの時系列 (膜電位・電流・スパイク) と STDP 窓の描画。

呼ぶのは `scripts/test.py` と `test/models/` の手動テスト、それに膜電位トレースを採る
実験 (`scripts/develop/panels.py`)。

保存先は `out_path` で受ける。`plotting/` の他の図と同じ約束で、**ファイル名を決めるのは
呼び出し側**。以前は `title` をファイル名に流用していたので、呼び出し側が
`PQN_test(..., title="test/models/neurons/PQN_test/PQN_V_test")` のようにタイトル欄へパスを
埋める羽目になっていた。`title` が残っているのは実際に図へ描く `stdp_window` だけ。
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


def neuron_trace(V, I_in, spike_times, spike_ids, dt, out_path, *, id=0, window_s=10.0):
    """
    単一ニューロンの膜電位・シナプス電流・スパイクを固定時間窓で描画する。

    neuron_test と同じレイアウト(ラスター / V / I)だが、x軸上限を config.task.duration
    ではなく引数 window_s で固定するため、長時間シミュレーション(72h等)から切り出した
    短い窓(既定10秒)をそのまま表示できる。

    Args:
        V: 対象ニューロンの膜電位 [mV] の1次元配列 (長さ = ステップ数)
        I_in: 対象ニューロンのシナプス電流 (Isyn) の1次元配列。None ならI パネルを省略
        spike_times: 窓内スパイク時刻 [ms] (窓の先頭を0とするローカル時刻)
        spike_ids: spike_times に対応するグローバルニューロンID
        dt: シミュレーションのタイムステップ [ms]
        out_path: 保存先ファイルパス
        id: 電圧トレースを表示する対象ニューロンID (ラスターの中心)
        window_s: x軸に表示する時間幅 [s]
    """
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

    save_figure(fig, out_path, dpi=None)
    plt.close()


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
