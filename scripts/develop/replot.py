"""既存の run から図と指標を作り直す (再解析)。

    python -m scripts.develop.replot <run ディレクトリ>

**本番と同じ関数を通る。** 記録窓ごとの出力は `report.panels.emit()`、run 全体は
`report.overview.emit()`、構造は `report.structure.emit()` で、どれも `run_one.py` が
呼ぶものと同一なので、列も値も図もバイト単位で一致する。

**作り直すのは全部。** 構造図も窓ごとの図も run 全体の図も、指標も、例外なく上書きする
(一部だけ新しい状態を作らないため)。

**構造図のためにネットワークを再ビルドする。** 構造図は「構築されたネットワークそのもの」
を見る図なので npz からは作れない。GeNN のコード生成とコンパイルは `sim.setup()` の側に
あり、`builder.build()` は走らせないので安い。ネットワークの生成は
`np.random.RandomState(config.simulation.seed)` だけに依存する (backend は GeNN の
デバイス RNG = スパイク列にしか効かない) ので、cuda で走らせた run でも cpu で再ビルド
できる。**再ビルドが元の run と同じ結合を作ったことは毎回確かめる**
(`BuiltNetwork.verify_against`)。
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.NetworkBuilder import NetworkBuilder
from src.core.output_manager import CONFIG_NAME, DATA_SUBDIR

from scripts.develop.report import overview, panels, structure
from scripts.develop.store import paths
from src.utils.runview import BuiltNetwork as Built
from scripts.develop.store.records import METRICS_NAME, MetricsWriter
from scripts.develop.store.series import open_run

# プロファイル名を Python クラスへ解決するために必要 (@register は import で走る)。
import src.models.neurons.akita_escape_lif
import src.models.neurons.akita_escape_lif_physical
import src.models.network.connectors
import src.models.network.delays
import src.models.network.space
import src.models.network.weights
import src.models.plasticity.custom_Akita
import src.models.synapses.standard_models
import src.models.synapses.custom

# 再ビルドに使うバックエンド。**構造は backend に依存しない**ので、CUDA の無い環境でも
# cuda の run を再解析できるよう cpu に固定する。一致は verify_against() が確かめる。
REBUILD_BACKEND = "cpu"
# 再ビルド用の GeNN モデル名。コンパイルはしないので <名前>_CODE は作られない。
REBUILD_MODEL_NAME = "develop_replot"


def parse_args():
    parser = argparse.ArgumentParser(description="既存 run の図と指標を作り直す。")
    parser.add_argument("run_dir", help="config.yaml と data/ を持つ run ディレクトリ")
    parser.add_argument("--no-structure", action="store_true",
                        help="構造図を飛ばす (再ビルドしない)。大きい run で時間を惜しむとき")
    return parser.parse_args()


def rebuild(series) -> Built:
    """`config.yaml` からネットワークを組み直し、記録と一致することを確かめる。"""
    config = series.config.model_copy(deep=True)
    config.simulation.backend = REBUILD_BACKEND
    builder = NetworkBuilder(config, model_name=REBUILD_MODEL_NAME)
    builder.build(rec_spike=True)
    built = Built.from_builder(builder, series.run_dir)
    built.verify_against(series.wiring())
    return built


def replot(run_dir: Path, *, with_structure: bool = True) -> Path:
    # <run>/data を渡されても run ルートに正規化する (タブ補完で入り込みやすいため)。
    run_dir = Path(run_dir)
    if run_dir.name == DATA_SUBDIR and (run_dir.parent / CONFIG_NAME).exists():
        run_dir = run_dir.parent

    series = open_run(run_dir)
    print(f"Replot: {run_dir} (N={series.total_neurons})")

    # 図のサブディレクトリを用意する。トレースは採った run にだけ作る
    # (空ディレクトリが残ると「採ったが空」と「採っていない」が読めなくなる)。
    fig_kinds = paths.DEFAULT_FIG_KINDS
    if any(window.trace_path is not None for window in series.windows):
        fig_kinds = fig_kinds + (paths.TRACE,)
    paths.prepare(run_dir, fig_kinds)

    if with_structure:
        print("\nRebuilding network for structure figures...")
        structure.emit(rebuild(series))

    metrics_csv = MetricsWriter(paths.data_path(run_dir, METRICS_NAME))
    for window in series.windows:
        panels.emit(window, metrics=metrics_csv)
    print(f"  Recomputed metrics -> {metrics_csv.path}")

    print("\nGenerating visualizations...")
    overview.emit(series)

    print(f"Replot done: {run_dir}")
    return run_dir


def main():
    args = parse_args()
    replot(Path(args.run_dir), with_structure=not args.no_structure)


if __name__ == "__main__":
    main()
