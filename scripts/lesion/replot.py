"""既存の損傷 run から図と指標を作り直す (再解析)。

    python -m scripts.lesion.replot <run ディレクトリ>

**本番と同じ関数を通る。** probe ごとの出力は `report.panels.emit()`、run 全体は
`report.overview.emit()`、切断後の構造は `report.structure.emit()` で、どれも
`run_one.py` が呼ぶものと同一。

**作り直すのは全部。** 構造図も probe ごとの図も run 全体の図も、指標も、例外なく
上書きする。一部だけ新しい状態を作らないため。

**構造図のためにネットワークを再ビルドして切り直し、重みを戻す。** 構造図は「切断後の
ネットワークそのもの」を見る図なので npz からは作れない。GeNN のコード生成とコンパイルは
`setup()` の側にあり `builder.build()` は走らせないので安い。

重みを戻すのを忘れると、config が生成した初期重み (この実験では全 0) のまま描くことに
なり、本番の構造図と食い違う —— しかも `network_sample.png` は重み 0 の結合を描かない
ので、**エッジが 1 本も無い図が黙って出る**。

切断マスクは `lesion_cut.npz` の (row, col) から**引き当てる**。位置で復元しないのは、
再ビルドした COO の並びが記録と同じであることを仮定しないため —— 仮定するのではなく、
引き当てに失敗したら止まる形にしてある。
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

from src.core.NetworkBuilder import NetworkBuilder
from src.core.output_manager import CONFIG_NAME, DATA_SUBDIR
from src.utils.analysis.axons import subset_geometry
from src.utils.analysis.weights import align_subset_to_coo
from src.utils.runview import BuiltNetwork

from scripts.lesion.report import overview, panels, structure
from scripts.lesion.store import paths
from scripts.lesion.store.records import METRICS_NAME, MetricsWriter
from scripts.lesion.store.series import open_run

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
# cuda の run を再解析できるよう cpu に固定する。
REBUILD_BACKEND = "cpu"
# 再ビルド用の GeNN モデル名。コンパイルはしないので <名前>_CODE は作られない。
REBUILD_MODEL_NAME = "lesion_replot"


def parse_args():
    parser = argparse.ArgumentParser(description="既存の損傷 run の図と指標を作り直す。")
    parser.add_argument("run_dir", help="config.yaml と data/ を持つ run ディレクトリ")
    parser.add_argument("--no-structure", action="store_true",
                        help="構造図を飛ばす (再ビルドしない)。大きい run で時間を惜しむとき")
    return parser.parse_args()


def cut_mask_for(coo, cut_row, cut_col) -> np.ndarray:
    """記録された切断シナプス (row, col) を、再ビルドした COO 上のマスクへ引き当てる。

    **位置ではなく (pre, post) の組で引く。** 位置で復元すると、並びが記録と 1 本でも
    ずれたときに黙って別のシナプスを切ることになる。引き当てに失敗したら止める ——
    失敗は「別のネットワークが出ている」ことの証拠なので、続行してよい状況が無い。
    """
    n_cols = int(coo.shape[1])
    keys = np.asarray(coo.row, dtype=np.int64) * n_cols + np.asarray(coo.col, dtype=np.int64)
    wanted = np.asarray(cut_row, dtype=np.int64) * n_cols + np.asarray(cut_col, dtype=np.int64)
    order = np.argsort(keys)
    position = np.searchsorted(keys[order], wanted)
    if np.any(position >= keys.size) or not np.array_equal(keys[order][np.clip(position, 0, keys.size - 1)], wanted):
        raise SystemExit(
            "lesion_cut.npz の切断シナプスを再ビルドしたネットワークに引き当てられません。"
            " 同じ seed から別のネットワークが出ています。構造図は描けません。")
    mask = np.zeros(keys.size, dtype=bool)
    mask[order[position]] = True
    return mask


def weights_at_cut(series, row, col) -> np.ndarray:
    """**切断の瞬間の**重みを、生き残ったシナプス `(row, col)` の上に並べ直す。

    構造図は重みを使う (ネットワーク図の線の太さ、重み分布のヒストグラム) ので、
    config が生成した初期重みのまま描くと本番と別の図になる。**本番が描いたのは
    Phase 2 へ流し込んだ重み** = baseline probe を測り終えた時点の値で、それは
    `weights_p000.npz` そのもの。

    引き当ては (pre, post) の join。baseline は切断前の全シナプスぶんあるので、
    位置では並ばない。
    """
    baseline = next((window for window in series.windows if window.is_baseline), None)
    if baseline is None:
        raise SystemExit(
            "baseline の probe がありません。切断の瞬間の重みが決まらないので、"
            " 構造図は描けません (--no-structure で飛ばせます)。")
    pre = series.wiring_pre()
    return align_subset_to_coo(pre.row, pre.col, baseline.weights(),
                               row, col, int(pre.shape[1]))


def rebuild(series) -> BuiltNetwork:
    """`config.yaml` からネットワークを組み直し、記録どおりに切って重みを戻す。"""
    config = series.config.model_copy(deep=True)
    config.simulation.backend = REBUILD_BACKEND
    builder = NetworkBuilder(config, model_name=REBUILD_MODEL_NAME)
    builder._generate_global_matrices()

    cut = series.cut()
    coo = builder.global_coo()
    keep = ~cut_mask_for(coo, cut["row"], cut["col"])

    geometry = getattr(builder.connection, "axon_geometry", lambda: None)()
    post_geometry = subset_geometry(geometry, keep) if geometry is not None else None

    row, col = coo.row[keep], coo.col[keep]
    builder.replace_global_coo(
        coo._replace(row=row, col=col, weights=weights_at_cut(series, row, col),
                     delays=coo.delays[keep]),
        preserve_fan_in_scale=True)

    built = BuiltNetwork.from_builder(builder, series.run_dir, geometry=post_geometry)
    built.verify_against(series.wiring())
    return built


def replot(run_dir: Path, *, with_structure: bool = True) -> Path:
    # <run>/data を渡されても run ルートに正規化する (タブ補完で入り込みやすいため)。
    run_dir = Path(run_dir)
    if run_dir.name == DATA_SUBDIR and (run_dir.parent / CONFIG_NAME).exists():
        run_dir = run_dir.parent

    series = open_run(run_dir)
    print(f"Replot: {run_dir} (N={series.total_neurons}, probe={len(series.windows)})")

    paths.prepare(run_dir)

    if with_structure:
        print("\nRebuilding lesioned network for structure figures...")
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
