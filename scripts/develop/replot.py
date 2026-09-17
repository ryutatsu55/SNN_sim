"""既存の run から図と指標を作り直す (再解析)。

    python -m scripts.develop.replot <run ディレクトリ> [--smax N]

**別コマンドにしてある。** 以前は `develop.py --replot-from` として同じスクリプトに同居して
いたが、そのせいで「1 記録窓の指標を作る」コードが本番用と再解析用の 2 つに分かれ、
再解析側にだけ E/I 列と重みブロック列が無い状態になっていた。いまはどちらも
`metrics.build_row()` を通るので、**同じ入力が揃っていれば列も一致する**。

GeNN は使わない (スパイクと重みの npz から計算し直すだけ)。

書き出し先は本番と同じ `metrics.csv`。**上書きしてよい。** 以前は別名 (`metrics_replot.csv`)
に逃がしていたが、それは再解析が本番より列の少ない CSV を作っていたからで、上書きすると
重みブロック列が失われるためだった。いまは同じ関数が同じ列を作るので、逃がす理由がない
(逃がしておく方がむしろ危険で、figure がどちらを読むかを呼び出し側が明示しないと
**例外を出さずに古い値を描く**)。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigManager
from src.core.layout import NetworkLayout
from src.core.output_manager import AXES_NAME, CONFIG_NAME, DATA_SUBDIR, locate, require

from scripts.develop import metrics, paths, panels
from scripts.develop.fig2c import plot_figure2c
from scripts.develop.fig2d import plot_figure2d
from scripts.develop.records import (METRICS_NAME, SPIKES, WEIGHTS, MetricsWriter,
                                    load_connectivity, load_weight_values,
                                    record_filename, discover_records)


def parse_args():
    parser = argparse.ArgumentParser(description="既存 run の図と指標を作り直す。")
    parser.add_argument("run_dir", help="config.yaml と spikes_*h.npz を持つ run ディレクトリ")
    parser.add_argument("--smax", type=int, default=None,
                        help="べき乗フィット / ΔCr の上限。既定は config の simulation.N")
    return parser.parse_args()


def _load_layout(run_dir: Path, config):
    """保存物から NetworkLayout を復元する。

    `from_config` が config.yaml から自動軸 (population / mode / polarity) を決定論的に
    再構築し、config だけでは再導出できない外部軸 (layer / module …) は
    `layout_axes.npz` から読み戻す。GeNN のコンパイルは要らない。

    **復元できなければ落とす。** layout が無いと E/I 列が作れず、本番より列の少ない
    metrics.csv で上書きしてしまう。「並べ替えだけ諦めて続行」は、失われるのが図ではなく
    記録なので割に合わない。
    """
    layout = NetworkLayout.from_config(config)
    axes_path = locate(run_dir, AXES_NAME)
    if axes_path is not None:
        layout.load_axes_file(axes_path)
    return layout


def replot(run_dir: Path, smax_override: int | None = None) -> Path:
    # run ルートと <run>/data のどちらを渡されても run ルートに正規化する。
    run_dir = Path(run_dir)
    if run_dir.name == DATA_SUBDIR and (run_dir / CONFIG_NAME).exists():
        run_dir = run_dir.parent

    config = ConfigManager().load_resolved(require(run_dir, CONFIG_NAME))
    record_window_ms = float(config.task.record_window_ms)
    total_neurons = int(config.simulation.N)
    smax = metrics.resolve_avalanche_smax(config, smax_override)
    wmax = metrics.max_plasticity_weight(config)
    print(f"Replot: {run_dir} (N={total_neurons}, avalanche smax={smax})")

    source_dir = paths.records_dir(run_dir)
    spike_files = discover_records(source_dir, SPIKES)
    if not spike_files:
        raise FileNotFoundError(f"spikes_*h.npz が見つかりません: {source_dir}")

    layout = _load_layout(run_dir, config)
    order_axes = panels.resolve_order_axes(layout)
    # 無ければ例外。「結合が無い」と「重みが 0」を区別できない古い密形式の run は、
    # 列を減らして解析するのではなく再実行してもらう (load_connectivity がそう案内する)。
    connectivity = load_connectivity(source_dir)

    # 図のサブディレクトリを用意する (旧レイアウトの run にも新しい置き方で描く)。
    paths.prepare(run_dir)

    # **指標は figure が読む場所へ、figure が読む名前で置く。**
    metrics_csv = MetricsWriter(source_dir / METRICS_NAME)
    for item in spike_files:
        hour = item.hour
        spikes_npz = np.load(item.path)
        times, ids = spikes_npz["times"], spikes_npz["ids"]
        local_times = times - hour * 60.0 * 60.0 * 1000.0

        weights = load_weight_values(source_dir / record_filename(WEIGHTS, hour))

        metrics_csv.append(metrics.build_row(
            hour, local_times, ids,
            total_neurons=total_neurons, record_window_ms=record_window_ms, smax=smax,
            layout=layout, weights=weights,
            row=connectivity.row, col=connectivity.col, wmax=wmax,
        ))

        panels.draw_raster(run_dir, hour, local_times, ids, total_neurons,
                           layout=layout, order_axes=order_axes)
        panels.draw_avalanche(run_dir, hour, local_times, smax)

    print(f"  Recomputed metrics -> {metrics_csv.path}")

    overview = str(paths.fig_dir(run_dir, paths.OVERVIEW))
    print("\nGenerating visualizations...")
    try:
        print("  Figure 2c...")
        plot_figure2c(str(run_dir), layout, output_dir=overview,
                      metrics_name=METRICS_NAME, llr_smax=smax)
    except Exception as error:
        print(f"  Warning: Figure 2c generation failed: {error}")
    try:
        print("  Figure 2d...")
        plot_figure2d(str(run_dir), layout, output_dir=overview)
    except Exception as error:
        print(f"  Warning: Figure 2d generation failed: {error}")

    print(f"Replot done: {run_dir}")
    return run_dir


def main():
    args = parse_args()
    replot(Path(args.run_dir), smax_override=args.smax)


if __name__ == "__main__":
    main()
