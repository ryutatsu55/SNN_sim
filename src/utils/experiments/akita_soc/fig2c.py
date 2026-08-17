import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 単体実行 (python -m ... でない直接実行) でも src パッケージを解決できるようにする
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.config_manager import ConfigManager
from src.core.layout import NetworkLayout
from src.core.output_manager import AXES_NAME, CONFIG_NAME, data_dir, locate, require
from src.utils.experiments.akita_soc.runio import WEIGHTS, discover_records, load_weight_matrix


def load_weight_trajectories(folder_path, layout):
    """
    フォルダ内のnpzファイルを読み込み、各シナプスの時間ごとの重みの軌跡を抽出する。
    グローバル ID ベースで興奮性・抑制性を区別して処理する。
    戻り値: (時間配列, トラジェクトリ辞書)
    トラジェクトリ辞書の各要素は (時間の数, シナプスの数) の2次元配列。
    """
    records = discover_records(Path(folder_path), WEIGHTS)

    if not records:
        print("警告: フォルダ内に weights_*h.npz が見つかりません。")
        return None, None

    ids = layout.ids_by("polarity")
    exc_ids = ids["excitatory"]
    inh_ids = ids["inhibitory"]

    times = []
    traj_EE, traj_EI, traj_IE, traj_II = [], [], [], []

    for record in records:
        times.append(record.hour)
        # 密形式 (キー "weights") と COO 形式 (キー "data" + connectivity.npz) の両方を
        # 扱えるローダを使う。npz のキーを直接見て 2 次元前提で添字すると、疎経路で
        # 記録した run で落ちる。
        W = load_weight_matrix(record.path)

        # グローバル ID を使ってブロックを抽出
        traj_EE.append(W[np.ix_(exc_ids, exc_ids)].flatten())
        traj_EI.append(W[np.ix_(exc_ids, inh_ids)].flatten())
        traj_IE.append(W[np.ix_(inh_ids, exc_ids)].flatten())
        traj_II.append(W[np.ix_(inh_ids, inh_ids)].flatten())

    trajectories = {
        'W_EE': np.array(traj_EE),
        'W_EI': np.array(traj_EI),
        'W_IE': np.array(traj_IE),
        'W_II': np.array(traj_II)
    }

    return np.array(times), trajectories

def plot_figure2c(folder, layout, output_dir=None):
    if output_dir is None:
        output_dir = folder

    # organize_output() 後は csv/npz が data/ にあるので、読み込みはそちらを見る。
    source_dir = str(data_dir(folder))
    metrics_path = os.path.join(source_dir, 'metrics.csv')

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
    npz_times, trajectories = load_weight_trajectories(source_dir, layout=layout)

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

    # 保存物から NetworkLayout を復元する。config.yaml から自動軸 (population/mode/polarity)
    # を再構築し、config だけでは再導出できない外部軸 (layer/module …) は layout_axes.npz
    # から読み戻す。
    config = ConfigManager().load_resolved(require(args.folder, CONFIG_NAME))
    layout = NetworkLayout.from_config(config)
    axes_path = locate(args.folder, AXES_NAME)
    if axes_path is not None:
        layout.load_axes_file(axes_path)

    plot_figure2c(args.folder, layout, output_dir=output_dir)

if __name__ == "__main__":
    main()