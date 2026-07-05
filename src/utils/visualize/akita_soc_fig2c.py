import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

def extract_hour(filename):
    """ファイル名 (例: weights_6.0h.npz) から時間を抽出する"""
    match = re.search(r'weights_(\d+\.?\d*)h\.npz', os.path.basename(filename))
    if match:
        return float(match.group(1))
    return -1.0

def infer_group_ids(folder_path):
    """config.yaml からグローバル ID を復元（visualize_weight_matrix.py と同じロジック）"""
    config_path = os.path.join(folder_path, 'config.yaml')
    if not os.path.exists(config_path):
        print(f"警告: {config_path} が見つかりません。")
        return None, None

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    simulation = config.get('simulation', {})
    neurons = config.get('neurons', {})

    if not neurons or 'seed' not in simulation:
        print("警告: seed または neurons 設定が見つかりません。")
        return None, None

    total_neurons = int(simulation.get('N', sum(int(cfg['num']) for cfg in neurons.values())))
    seed = int(simulation['seed'])
    rng = np.random.RandomState(seed)

    available_indices = np.arange(total_neurons)
    excitatory_ids = []
    inhibitory_ids = []

    for cfg in neurons.values():
        num_neurons = int(cfg['num'])
        assigned = rng.choice(available_indices, size=num_neurons, replace=False)
        available_indices = np.setdiff1d(available_indices, assigned)
        assigned.sort()
        mode = cfg.get('mode') or ''
        if mode.startswith('excitatory'):
            excitatory_ids.append(assigned)
        elif mode.startswith('inhibitory'):
            inhibitory_ids.append(assigned)

    exc_ids = np.concatenate(excitatory_ids) if excitatory_ids else np.array([], dtype=np.int32)
    inh_ids = np.concatenate(inhibitory_ids) if inhibitory_ids else np.array([], dtype=np.int32)

    return exc_ids, inh_ids

def load_weight_trajectories(folder_path, layout=None):
    """
    フォルダ内のnpzファイルを読み込み、各シナプスの時間ごとの重みの軌跡を抽出する。
    グローバル ID ベースで興奮性・抑制性を区別して処理する。
    戻り値: (時間配列, トラジェクトリ辞書)
    トラジェクトリ辞書の各要素は (時間の数, シナプスの数) の2次元配列。
    """
    npz_files = glob.glob(os.path.join(folder_path, 'weights_*h.npz'))
    npz_files.sort(key=extract_hour)

    if not npz_files:
        print("警告: フォルダ内に weights_*h.npz が見つかりません。")
        return None, None

    # layout(NetworkLayout)が与えられた場合はそれを使用
    if layout is not None:
        ids = layout.ids_by_mode()
        exc_ids = ids["excitatory"]
        inh_ids = ids["inhibitory"]
    else:
        exc_ids, inh_ids = infer_group_ids(folder_path)
        if exc_ids is None or inh_ids is None:
            return None, None

    times = []
    traj_EE, traj_EI, traj_IE, traj_II = [], [], [], []

    for file in npz_files:
        times.append(extract_hour(file))
        with np.load(file) as data:
            array_key = data.files[0]
            W = data[array_key]

            # グローバル ID を使ってブロックを抽出
            W_EE = W[np.ix_(exc_ids, exc_ids)]
            W_EI = W[np.ix_(exc_ids, inh_ids)]
            W_IE = W[np.ix_(inh_ids, exc_ids)]
            W_II = W[np.ix_(inh_ids, inh_ids)]

            traj_EE.append(W_EE.flatten())
            traj_EI.append(W_EI.flatten())
            traj_IE.append(W_IE.flatten())
            traj_II.append(W_II.flatten())

    trajectories = {
        'W_EE': np.array(traj_EE),
        'W_EI': np.array(traj_EI),
        'W_IE': np.array(traj_IE),
        'W_II': np.array(traj_II)
    }

    return np.array(times), trajectories

def plot_figure2c(folder, output_dir=None, layout=None):
    if output_dir is None:
        output_dir = folder

    metrics_path = os.path.join(folder, 'metrics.csv')

    if not os.path.exists(metrics_path):
        print(f"エラー: {metrics_path} が見つかりません。")
        return

    # ==========================================
    # データ読み込み
    # ==========================================
    # 右列用: csvの読み込み
    df = pd.read_csv(metrics_path)
    csv_times = df['hour'].values

    # 左列用: npzファイル群からのトラジェクトリデータ生成
    npz_times, trajectories = load_weight_trajectories(folder, layout=layout)

    # ==========================================
    # グラフの描画設定
    # ==========================================
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex='col')
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    # 左列: シナプス重みの時間発展（個別の線の束）
    if trajectories:
        weight_types = ['W_EE', 'W_EI', 'W_IE', 'W_II']
        
        # 線の透明度。濃すぎる場合は数値を下げ、薄すぎる場合は上げる（例: 0.01 ~ 0.05）
        line_alpha = 0.02 
        
        for i, w_type in enumerate(weight_types):
            ax = axes[i, 0]
            
            # plot関数に X(1次元), Y(2次元) を渡すと、Yの列数分の線が同時にプロットされる
            ax.plot(npz_times, trajectories[w_type], color='black', alpha=line_alpha, linewidth=0.5)
            
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_ylabel(w_type)
            
            if i == 0:
                ax.set_title('Synaptic Weights Development')

    # 右列: ネットワーク指標の推移
    metrics_cols = [
        ('llr', 'LLR'),
        ('bimodality_d', 'D'),
        ('delta_cr', 'ΔCr'),
        ('burstiness_index', 'BI')
    ]

    for i, (col, ylabel) in enumerate(metrics_cols):
        ax = axes[i, 1]
        if col in df.columns:
            ax.plot(csv_times, df[col], color='black', linewidth=1.5)
            
            if col == 'delta_cr':
                ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            elif col == 'burstiness_index':
                y_max = max(df[col]) if max(df[col]) > 0 else 0.5
                ax.set_ylim(0, y_max * 1.2)
                
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            if i == 0:
                ax.set_title('Network Characteristics')

    # 軸ラベルの設定
    axes[3, 0].set_xlabel('Time (h)')
    axes[3, 1].set_xlabel('Time (h)')
    
    # 左右のx軸の範囲を揃える
    max_time = max(csv_times.max() if len(csv_times) > 0 else 0, 
                   max(npz_times) if npz_times is not None else 0)
    axes[3, 0].set_xlim(0, max_time)
    axes[3, 1].set_xlim(0, max_time)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'figure2c_reproduction.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"グラフを {output_file} に保存しました。")
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description="論文Fig.2(c)の再現プロット生成")
    parser.add_argument("folder", type=str, help="metrics.csvとweights_*.npzが含まれるフォルダパス")
    parser.add_argument("--output-dir", default=None, help="出力ディレクトリ。未指定なら folder と同じ場所")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.folder
    plot_figure2c(args.folder, output_dir)

if __name__ == "__main__":
    main()