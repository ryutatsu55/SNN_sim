import os
import sys
import glob
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml

# 単体実行 (python -m ... でない直接実行) でも src パッケージを解決できるようにする
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.akita_soc import firing_rates
from src.utils.visualize.akita_soc_fig2c import infer_group_ids


def extract_hour(filename):
    """ファイル名 (例: spikes_6.0h.npz) から時間を抽出する"""
    match = re.search(r'spikes_(\d+\.?\d*)h\.npz', os.path.basename(filename))
    if match:
        return float(match.group(1))
    return -1.0


def _resolve_data_folder(folder):
    """spikes_*h.npz が入っているフォルダを解決する。

    指定 folder 直下になければ organize_output が作る folder/data/ を確認する。
    どちらにも無ければ元の folder を返す (呼び出し側で警告)。
    """
    if glob.glob(os.path.join(folder, 'spikes_*h.npz')):
        return folder
    data_dir = os.path.join(folder, 'data')
    if glob.glob(os.path.join(data_dir, 'spikes_*h.npz')):
        return data_dir
    return folder


def resolve_group_ids(folder, layout=None):
    """興奮性・抑制性のグローバル ID を取得する。

    layout(NetworkLayout) が与えられた場合はそれを使用し、
    無ければ config.yaml から infer_group_ids() で復元する。
    """
    if layout is not None:
        ids = layout.ids_by_mode()
        return np.sort(ids["excitatory"]), np.sort(ids["inhibitory"])

    return infer_group_ids(folder)


def load_firing_rate_series(folder, layout=None):
    """spikes_*h.npz 群から各時刻のニューロン別発火レートを収集する。

    戻り値:
        times: shape [T] の時刻配列 (h)
        rates: shape [T, N] のニューロン別発火レート (Hz)
        exc_ids, inh_ids: 興奮性・抑制性のグローバル ID
    """
    npz_files = glob.glob(os.path.join(folder, 'spikes_*h.npz'))
    npz_files.sort(key=extract_hour)

    if not npz_files:
        print(f"警告: フォルダ内に spikes_*h.npz が見つかりません: {folder}")
        return None, None, None, None

    config_path = os.path.join(folder, 'config.yaml')
    if not os.path.exists(config_path):
        print(f"警告: {config_path} が見つかりません。")
        return None, None, None, None

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    simulation = config.get('simulation', {})
    neurons = config.get('neurons', {})
    total_neurons = int(simulation.get('N', sum(int(cfg['num']) for cfg in neurons.values())))
    record_window_ms = float(config.get('task', {}).get('record_window_ms', 0.0))
    if record_window_ms <= 0:
        print(f"警告: record_window_ms が不正です ({record_window_ms})。")
        return None, None, None, None

    exc_ids, inh_ids = resolve_group_ids(folder, layout=layout)
    if exc_ids is None or inh_ids is None:
        return None, None, None, None

    times = []
    rates = []
    for file in npz_files:
        times.append(extract_hour(file))
        with np.load(file) as data:
            ids = data['ids']
        rates.append(firing_rates(ids, total_neurons, record_window_ms))

    return np.array(times), np.array(rates), exc_ids, inh_ids


def plot_figure2d(folder, output_dir=None, layout=None):
    """個々のニューロンの発火レート推移を散布図で描画する (興奮性=赤, 抑制性=青)。"""
    data_folder = _resolve_data_folder(folder)
    if output_dir is None:
        output_dir = folder

    times, rates, exc_ids, inh_ids = load_firing_rate_series(data_folder, layout=layout)
    if times is None or rates is None:
        print("エラー: 発火レートデータを収集できませんでした。")
        return

    # 各時刻 h における全ニューロンのレートを (x=h, y=rate) の点として展開する。
    n_neurons = rates.shape[1]
    t_grid = np.repeat(times[:, None], n_neurons, axis=1)  # shape [T, N]

    fig, ax = plt.subplots(figsize=(10, 6))

    for ids, color, label in ((inh_ids, 'blue', 'Inhibitory'),
                              (exc_ids, 'red', 'Excitatory')):
        if ids is None or ids.size == 0:
            continue
        ids = ids[ids < n_neurons]
        ax.scatter(
            t_grid[:, ids].ravel(),
            rates[:, ids].ravel(),
            s=6,
            c=color,
            alpha=0.35,
            edgecolors='none',
            label=label,
        )

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title('Per-neuron firing rate development')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', markerscale=2.0)
    if times.size > 0:
        ax.set_xlim(times.min() - 0.5, times.max() + 0.5)
    ax.set_ylim(bottom=0)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'figure2d_firing_rate_scatter.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"グラフを {output_file} に保存しました。")


def main():
    parser = argparse.ArgumentParser(description="個々のニューロン発火レート推移の散布図を生成 (Fig.2 d)")
    parser.add_argument("folder", type=str, help="spikes_*h.npz と config.yaml が含まれるフォルダパス")
    parser.add_argument("--output-dir", default=None, help="出力ディレクトリ。未指定なら folder と同じ場所")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.folder
    plot_figure2d(args.folder, output_dir)


if __name__ == "__main__":
    main()
