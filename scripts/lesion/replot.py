"""既存の損傷 run から図と指標を作り直す (再解析)。

    python -m scripts.lesion.replot <run ディレクトリ> [--verify]

**本番と同じ関数を通る。** probe ごとの出力は `report.panels.emit()`、run 全体は
`report.overview.emit()`、切断後の構造は `report.structure.emit()` で、どれも
`run_one.py` が呼ぶものと同一。

**作り直すのは全部。** 構造図も probe ごとの図も run 全体の図も、指標も、例外なく
上書きする。一部だけ新しい状態を作らないため。

**構造図も記録から描く。** 切断直後の記録 (`store/build_record.py`: 切断後の結合・遅延・
重み・軸索幾何) を `series.built()` で読むので、本番の run と同じ呼び出しになる。
ネットワークは組み直さない。

`--verify` を付けたときだけ、`config.yaml` からネットワークを**再ビルド**し、記録どおりに
切って、記録と同じ結合集合 (と軸索幾何) が出るかを確かめる。切断マスクは
`lesion_cut.npz` の (row, col) から**引き当てる** —— 位置で復元すると、並びが 1 本でも
ずれたときに黙って別のシナプスを切ることになる。ふだんの守りは golden テスト
(`test/test_network_golden.py`)。
"""
import argparse
import dataclasses
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.NetworkBuilder import NetworkBuilder
from src.utils.runview import BuiltNetwork, optional

from scripts.lesion.analysis.restore import verify_axon_geometry
from scripts.lesion.report import overview, panels, structure
from scripts.lesion.store import paths
from scripts.lesion.store.records import METRICS_NAME, MetricsWriter
from scripts.lesion.store.views import open_run

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

# `--verify` の再ビルドに使うバックエンド。**構造は backend に依存しない**ので、CUDA の
# 無い環境でも cuda の run を確かめられるよう cpu に固定する。
REBUILD_BACKEND = "cpu"
# 再ビルド用の GeNN モデル名。コンパイルはしないので <名前>_CODE は作られない。
REBUILD_MODEL_NAME = "lesion_replot"


def parse_args():
    parser = argparse.ArgumentParser(description="既存の損傷 run の図と指標を作り直す。")
    parser.add_argument("run_dir", help="config.yaml と data/ を持つ run ディレクトリ")
    parser.add_argument("--no-structure", action="store_true",
                        help="構造図を飛ばす")
    parser.add_argument("--verify", action="store_true",
                        help="config.yaml から再ビルドして切り直し、記録と同じ結合が出るか確かめる")
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
            " 同じ seed から別のネットワークが出ています。")
    mask = np.zeros(keys.size, dtype=bool)
    mask[order[position]] = True
    return mask


def verify_rebuild(series) -> None:
    """`config.yaml` からネットワークを組み直して記録どおりに切り、記録と一致するか確かめる。

    結合は (row, col) の集合として、軸索幾何は記録があれば配列ごと比べる。違えば止める。
    """
    config = series.config.model_copy(deep=True)
    config.simulation.backend = REBUILD_BACKEND
    builder = NetworkBuilder(config, model_name=REBUILD_MODEL_NAME)
    builder._generate_global_matrices()

    cut = series.cut()
    coo = builder.global_coo()
    keep = ~cut_mask_for(coo, cut["row"], cut["col"])
    builder.replace_global_coo(
        coo._replace(row=coo.row[keep], col=coo.col[keep], weights=coo.weights[keep],
                     delays=coo.delays[keep]),
        preserve_fan_in_scale=True)
    BuiltNetwork.from_builder(builder, series.run_dir).verify_against(series.wiring())

    geometry = getattr(builder.connection, "axon_geometry", lambda: None)()
    recorded = optional(series.geometry)
    if geometry is not None and recorded is not None:
        rebuilt = dataclasses.replace(
            geometry,
            pre=geometry.pre[keep], post=geometry.post[keep],
            contact_seg=geometry.contact_seg[keep], contact_t=geometry.contact_t[keep])
        verify_axon_geometry(rebuilt, recorded)
    print("  再ビルドして切り直した結合は記録と一致しました。")


def replot(run_dir: Path, *, with_structure: bool = True, verify: bool = False) -> Path:
    run_dir = Path(run_dir)
    series = open_run(run_dir)
    print(f"Replot: {run_dir} (N={series.total_neurons}, probe={len(series.windows)})")

    paths.prepare(run_dir)

    if verify:
        print("\nRebuilding lesioned network to verify it against the record...")
        verify_rebuild(series)

    if with_structure:
        print("\nGenerating post-lesion structure figures...")
        structure.emit(series.built())

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
    replot(Path(args.run_dir), with_structure=not args.no_structure, verify=args.verify)


if __name__ == "__main__":
    main()
