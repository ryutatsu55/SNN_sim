import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml

# 単体実行 (python -m ... でない直接実行) でも src パッケージを解決できるようにする
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.analysis.spikes import firing_rates
from src.utils.experiments.akita_soc.runio import SPIKES, discover_records
from src.core.config_manager import ConfigManager
from src.core.layout import NetworkLayout
from src.core.output_manager import AXES_NAME, CONFIG_NAME, data_dir, locate, require


def load_firing_rate_series(folder, layout):
    """spikes_*h.npz 群から各時刻のニューロン別発火レートを収集する。

    戻り値:
        times: shape [T] の時刻配列 (h)
        rates: shape [T, N] のニューロン別発火レート (Hz)
        exc_ids, inh_ids: 興奮性・抑制性のグローバル ID
    """
    records = discover_records(Path(folder), SPIKES)

    if not records:
        print(f"警告: フォルダ内に spikes_*h.npz が見つかりません: {folder}")
        return None, None, None, None

    config_path = locate(folder, CONFIG_NAME)
    if config_path is None:
        print(f"警告: {CONFIG_NAME} が見つかりません: {folder}")
        return None, None, None, None

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    record_window_ms = float(config.get('task', {}).get('record_window_ms', 0.0))
    if record_window_ms <= 0:
        print(f"警告: record_window_ms が不正です ({record_window_ms})。")
        return None, None, None, None

    polarity_ids = layout.ids_by("polarity")
    exc_ids, inh_ids = polarity_ids["excitatory"], polarity_ids["inhibitory"]
    total_neurons = layout.total_neurons

    times = []
    rates = []
    for record in records:
        times.append(record.hour)
        with np.load(record.path) as data:
            ids = data['ids']
        rates.append(firing_rates(ids, total_neurons, record_window_ms))

    return np.array(times), np.array(rates), exc_ids, inh_ids


def plot_figure2d(folder, layout, output_dir=None):
    """個々のニューロンの発火レート推移を散布図で描画する (興奮性=赤, 抑制性=青)。"""
    data_folder = str(data_dir(folder))
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

    # 保存物から NetworkLayout を復元する。config.yaml から自動軸 (population/mode/polarity)
    # を再構築し、config だけでは再導出できない外部軸 (layer/module …) は layout_axes.npz
    # から読み戻す。
    config = ConfigManager().load_resolved(require(args.folder, CONFIG_NAME))
    layout = NetworkLayout.from_config(config)
    axes_path = locate(args.folder, AXES_NAME)
    if axes_path is not None:
        layout.load_axes_file(axes_path)

    plot_figure2d(args.folder, layout, output_dir=output_dir)


if __name__ == "__main__":
    main()
