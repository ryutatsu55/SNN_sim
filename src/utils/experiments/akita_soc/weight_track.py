"""重み行列の時間発展を可視化する。

グローバル ID ベースで興奮性・抑制性のブロック構造を表示し、ブロック別統計を
weight_block_metrics.csv に書き出す。
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

# 単体実行 (python -m ... でない直接実行) でも src パッケージを解決できるようにする
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.config_manager import ConfigManager
from src.core.layout import NetworkLayout
from src.core.output_manager import AXES_NAME, CONFIG_NAME, data_dir, locate, require
from src.utils.analysis.weights import compute_block_metrics
from src.utils.experiments.akita_soc.runio import (
    WEIGHTS,
    connection_mask_from_config,
    discover_records,
    load_weight_matrix,
    write_metrics_csv,
)
from src.utils.plotting.matrices import plot_single_weight_matrix, plot_weight_panel
from src.utils.plotting.ordering import DEFAULT_ORDER_AXES


def visualize_weight_tracks(
    run_dir: Path,
    layout,
    output_dir: Path | None = None,
    order_axes: tuple[str, ...] | None = DEFAULT_ORDER_AXES,
) -> Path:
    """
    重み行列の時間発展を可視化する。

    Args:
        run_dir: weights_*h.npz を含む実験ディレクトリ
        layout: NetworkLayout。ディスクからは読まないので呼び出し側が用意すること
                (CLI から実行する場合は main() が config.yaml から復元する)
        output_dir: 出力先ディレクトリ。None の場合は run_dir
        order_axes: 重み行列の並べ替えに使う軸を外側から順に (既定は E/I でブロック化)。
                    None ならグローバル ID の順のまま。

    Returns:
        出力ディレクトリのパス
    """
    run_dir = run_dir.resolve()
    if output_dir is None:
        output_dir = run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # organize_output() 後は npz が data/ にあるので、列挙はそちらを見る (出力は run 直下)。
    weight_files = discover_records(data_dir(run_dir), WEIGHTS)
    if not weight_files:
        raise FileNotFoundError(f"No weights_*h.npz files found in: {run_dir}")

    first_weights = load_weight_matrix(weight_files[0].path)

    if layout.total_neurons != first_weights.shape[0]:
        raise ValueError(
            f"Layout total_neurons={layout.total_neurons} does not match weight matrix size={first_weights.shape[0]}."
        )

    connection_mask = connection_mask_from_config(run_dir, first_weights.shape[0])
    weight_items = []
    delta_items = []
    metric_rows = []
    previous_weights = None

    for item in weight_files:
        weights = load_weight_matrix(item.path)
        if weights.shape != first_weights.shape:
            raise ValueError(f"Weight shape mismatch: {item.path}")
        weight_items.append((item.hour, weights))
        metric_rows.extend(
            compute_block_metrics(
                hour=item.hour,
                weights=weights,
                layout=layout,
                connection_mask=connection_mask,
                previous_weights=previous_weights,
            )
        )
        plot_single_weight_matrix(
            weights=weights,
            layout=layout,
            out_path=output_dir / f"weight_matrix_{item.hour:g}h.png",
            title=f"Weight matrix {item.hour:g} h",
            order_axes=order_axes,
        )
        if previous_weights is not None:
            delta_items.append((item.hour, weights - previous_weights))
        previous_weights = weights

    plot_weight_panel(
        weight_items=weight_items,
        layout=layout,
        out_path=output_dir / "weight_matrix_panel.png",
        title=f"Weight matrix timeline: {run_dir.name}",
        order_axes=order_axes,
    )
    if delta_items:
        plot_weight_panel(
            weight_items=delta_items,
            layout=layout,
            out_path=output_dir / "weight_delta_panel.png",
            title=f"Weight delta from previous record: {run_dir.name}",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            order_axes=order_axes,
        )

    write_metrics_csv(metric_rows, output_dir / "weight_block_metrics.csv")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重み行列の時間発展を可視化します。")
    parser.add_argument("run_dir", nargs="+", help="weights_*h.npz を含む実験ディレクトリ")
    parser.add_argument("--output-dir", default=None, help="本スクリプト実行時の出力先。未指定の場合 <run_dir>/")
    parser.add_argument(
        "--order-axes", nargs="+", default=list(DEFAULT_ORDER_AXES), metavar="AXIS",
        help="重み行列を並べ替える軸を外側から順に (既定: polarity)。"
             " 例: --order-axes layer polarity。'none' でグローバル ID の順のまま",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if output_dir is not None and len(args.run_dir) > 1:
        print("--output-dir は run_dir が1つのときだけ指定できます。", file=sys.stderr)
        return 2

    order_axes = None if args.order_axes == ["none"] else tuple(args.order_axes)

    for run_dir_arg in args.run_dir:
        run_dir = Path(run_dir_arg)
        # 保存物から NetworkLayout を復元する。config.yaml から自動軸 (population/mode/
        # polarity) を再構築し、config だけでは再導出できない外部軸 (layer/module …) は
        # layout_axes.npz から読み戻す。
        config = ConfigManager().load_resolved(require(run_dir, CONFIG_NAME))
        layout = NetworkLayout.from_config(config)
        axes_path = locate(run_dir, AXES_NAME)
        if axes_path is not None:
            layout.load_axes_file(axes_path)

        try:
            out_dir = visualize_weight_tracks(run_dir, layout, output_dir=output_dir,
                                              order_axes=order_axes)
        except KeyError as error:
            # 並べ替え軸の指定ミス。保存済みの外部軸は run ごとに違うので、素の
            # トレースバックではなく「その run で使える軸」を見せる。
            print(f"--order-axes を解決できません: {error}", file=sys.stderr)
            return 2
        print(f"Weight matrix visualizations saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
