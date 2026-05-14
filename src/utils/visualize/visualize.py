import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def export_spike_csv(spike_time: np.ndarray, neuron_id: np.ndarray, output_path="spikes.csv"):
    """発火時刻とニューロンIDをCSVとして保存する関数"""

    spike_time = np.asarray(spike_time, dtype=float).reshape(-1)
    neuron_id = np.asarray(neuron_id).reshape(-1)

    if spike_time.size == 0 or neuron_id.size == 0:
        raise ValueError("spike_time and neuron_id arrays must not be empty.")
    if spike_time.shape[0] != neuron_id.shape[0]:
        raise ValueError("spike_time and neuron_id arrays must have the same length.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spike_table = np.column_stack((spike_time, neuron_id.astype(int, copy=False)))
    np.savetxt(
        output_path,
        spike_table,
        delimiter=",",
        header="spike_time,neuron_id",
        comments="",
        fmt=["%.10f", "%d"],
    )

def raster(times, ids, title="Raster Plot", tmax=None, idmax=None, s=10.0, save_path="."):
    """
    スパイク時刻とニューロンIDの配列からラスタープロットを作成する。
    
    Parameters:
        times (np.ndarray): スパイクが発生した時刻の1次元配列[ms]
        ids (np.ndarray): スパイクを発火したニューロンIDの1次元配列
        title (str): グラフのタイトル
        tmax (float): 時間軸の最大値。Noneの場合は自動
        idmax (float): Y軸(ID)の最大値。Noneの場合は自動
        s (float): マーカーのサイズ
        save_path (str): 保存先のパス。Noneの場合は画面に表示
    """
    times = times/1000.0  # ms -> s に変換
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(times, ids, s=s)
    
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Neuron ID")
    
    # 表示範囲の指定がある場合
    if tmax is not None:
        ax.set_xlim(0, tmax)
    if idmax is not None:
        ax.set_ylim(0, idmax)

    plt.tight_layout()
    
    plt.savefig(f"{save_path}/{title}", dpi=300, bbox_inches='tight')
    print(f"Raster plot saved to {save_path}")
        
    plt.close()


def PQN_test(V_data, I_in, config, title="PQN_V_test"):
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
    
    plt.tight_layout()
    plt.savefig(f"{title}.png")

def neuron_test(V_data, I_in, spike_times, spike_ids, config, id=0, title="neuron_test",save_path="."):
    V_data = V_data[:, id]
    I_in = I_in[:,id]
    tmax = config.task.duration/1000
    time_axis = np.arange(len(V_data)) * config.simulation.dt / 1000.0
    
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 4, 1]}, figsize=(9, 4), sharex=True)
    
    spike_times = spike_times/1000.0
    ax0.scatter(spike_times, spike_ids, s=10)
    ax0.set_ylabel("Neuron ID")
    ax0.set_ylim(id-0.5, id+0.5)

    ax1.plot(time_axis, V_data, color='tab:blue')
    ax1.set_ylabel('v [mV]')
    ax1.set_xlim(0, tmax)
    
    ax2.plot(time_axis, I_in, color='black')
    ax2.set_ylabel('I [nA]')
    ax2.set_xlabel('Time [s]')
    ax2.set_xlim(0, tmax)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{title}.png")

def network(weights: np.ndarray, coords: np.ndarray, config, node_size=10, title="network", save_path="."):
    """
    ニューロンの空間配置と重み行列からネットワーク構造を可視化する。
    
    Parameters:
        coords (np.ndarray): ニューロンの座標配列。形状は (N, 3)。
        weights (np.ndarray): 結合重み行列。形状は (N, N)。
        node_size (int): ニューロン(ノード)の描画サイズ。
        title (str): グラフのタイトル。
        save_path (str or None): 画像の保存先パス。Noneの場合は画面に表示する。
    """
    N = weights.shape[0]
    
    # Z軸が存在する場合でも、今回は2D平面(X, Y)への投影として扱う
    x = coords[:, 0]
    y = coords[:, 1]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(x, y, s=node_size, color='darkgray', edgecolors='black', zorder=3)
    
    # 描画用のスケール計算（太さの正規化用）
    max_weight = np.max(np.abs(weights)) if np.max(np.abs(weights)) > 0 else 1.0
    # 結合（エッジ）を抽出
    sources, targets = np.where(np.abs(weights) != 0)

    # 矢印がノードの中心に刺さるのを防ぐためのマージン計算
    # (scatterの s は面積なので、半径は平方根に比例)
    node_margin = np.sqrt(node_size) * 0.8
    
    for s, t in zip(sources, targets):
        
        # 重みの強さに応じて線の太さを変更 (最大2.0)
        w = weights[s, t]
        lw = (abs(w) / max_weight) * 2.0
        
        # 正(興奮性)は赤系、負(抑制性)は青系に色分け
        if w > 0:
            color = (0.8, 0.2, 0.2, 0.5)  # RGBA (Red, alpha=0.5)
        else:
            color = (0.2, 0.2, 0.8, 0.5)  # RGBA (Blue, alpha=0.5)

        # ax.annotate を用いて矢印を描画
        ax.annotate(
            "", 
            xy=(x[t], y[t]),       # 終点 (Target)
            xytext=(x[s], y[s]),   # 始点 (Source)
            arrowprops=dict(
                arrowstyle="->, head_length=0.4, head_width=0.2", # 矢印の形状
                color=color,
                linewidth=lw,
                shrinkA=node_margin,  # 始点側の隙間（ノードと重ならないように）
                shrinkB=node_margin,  # 終点側の隙間（矢印の先がノードに隠れないように）
                # 双方向の結合が重ならないよう、線を少しカーブさせる (rad=0.1)
                connectionstyle="arc3,rad=0.1" 
            ),
            zorder=1
        )
    
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel("X Coordinate [um]")
    ax.set_xlim(config.network.space.x_range)
    ax.set_ylabel("Y Coordinate [um]")
    ax.set_ylim(config.network.space.y_range)

    plt.tight_layout()
    
    plt.savefig(f"{save_path}/{title}.png", dpi=300, bbox_inches='tight')
    print(f"Network visualization saved to {save_path}/{title}.png")
        
    plt.close()

def stdp_window(dw: np.ndarray, dt: np.ndarray, title="stdp_window", save_path="."):
    """
    STDPの学習特性（Δt vs Δw）をプロットし、画像として保存する関数。
    
    Args:
        dw: 重みの変化量 (Δw = w_after - w_before) の配列
        dt: スパイク時間差 (Δt = t_post - t_pre) [ms] の配列
        title: グラフ/ファイル名
        save_path: 画像の保存先ディレクトリ
    """
    plt.figure(figsize=(10, 6))
    
    # 0点を強調するガイドライン
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.axvline(0, color='black', linewidth=1, linestyle='--')
    
    # データの散布図と近似線
    plt.scatter(dt, dw, color='blue', alpha=0.6, label='Measured Data')
    
    # スムーズな曲線を描くためのソート処理（プロット用）
    sort_idx = np.argsort(dt)
    plt.plot(dt[sort_idx], dw[sort_idx], color='red', linewidth=2, label='STDP Curve')
    
    # 軸ラベルの設定
    plt.xlabel('Spike Timing Difference: $\Delta t$ [ms]', fontsize=12)
    plt.ylabel('Weight Change: $\Delta w$ (or $\Delta g$)', fontsize=12)
    plt.title(title, fontsize=14)
    
    # 領域の解説（LTP/LTD）
    plt.text(max(dt)*0.7, max(dw)*0.1, 'LTP (Potentiation)', fontsize=10, color='green', fontweight='bold')
    plt.text(min(dt)*0.7, max(dw)*0.1, 'LTD (Depression)', fontsize=10, color='orange', fontweight='bold')
    
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend()
    
    # 保存処理
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    file_full_path = os.path.join(save_path, f"{title}.png")
    plt.savefig(file_full_path, dpi=300)
    plt.close()
    
    print(f"  [Visualization] STDP window plot saved to: {file_full_path}")
