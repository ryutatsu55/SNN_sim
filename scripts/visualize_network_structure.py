"""YAML 設定からネットワークを構築し、構造図一式を出力する。

入力は **run ディレクトリではなく config** で、NetworkBuilder を実際に走らせる
(GeNN のコンパイルと実行は行わず、呼ぶのは `_generate_global_matrices()` だけ)。
既存の実験結果を解析する `src/utils/experiments/` の CLI とは入力が別物なので、
実験を回す側である scripts/ に置いている。

    python scripts/visualize_network_structure.py configs/test.yaml -o output/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# プロジェクトルートにパスを通す
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.plotting.network import (
    network,
    plot_connection_mask_coarse,
    plot_delay_distribution,
    plot_empirical_connection_probability,
    plot_weight_distributions,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _coo_from_builder(builder):
    """NetworkBuilder の生成結果を COO (row, col, weights, delays[ms]) に正規化する。

    疎経路なら sparse_* をそのまま、密経路ならマスクの非零要素を抜き出す。
    """
    if builder.is_sparse:
        return (
            np.asarray(builder.sparse_rows, dtype=np.int64),
            np.asarray(builder.sparse_cols, dtype=np.int64),
            np.asarray(builder.sparse_weights, dtype=np.float64),
            np.asarray(builder.sparse_delays, dtype=np.float64),
        )
    mask = np.asarray(builder.global_mask)
    row, col = np.nonzero(mask)
    return (
        row.astype(np.int64),
        col.astype(np.int64),
        np.asarray(builder.global_weights)[row, col].astype(np.float64),
        np.asarray(builder.global_delays)[row, col].astype(np.float64),
    )


def _dense_weights(builder):
    """空間ネットワーク図 (plotting.network.network) 用に密な重み行列 (N, N) を得る。

    密経路なら global_weights をそのまま、疎経路なら COO から復元する。
    """
    if builder.global_weights is not None:
        return np.asarray(builder.global_weights)
    total = builder.total_neurons
    W = np.zeros((total, total), dtype=np.float64)
    W[np.asarray(builder.sparse_rows, dtype=np.int64),
      np.asarray(builder.sparse_cols, dtype=np.int64)] = builder.sparse_weights
    return W


def _register_plugins():
    """@register デコレータを走らせてコンポーネントをレジストリに登録する。"""
    import src.models.neurons.pqn_float  # noqa: F401
    import src.models.neurons.pqn_int  # noqa: F401
    import src.models.neurons.akita_escape_lif  # noqa: F401
    import src.models.neurons.akita_escape_lif_physical  # noqa: F401
    import src.models.neurons.lif  # noqa: F401
    import src.models.network.space  # noqa: F401
    import src.models.network.connectors  # noqa: F401
    import src.models.network.weights  # noqa: F401
    import src.models.network.delays  # noqa: F401


def _default_task(active_task: str | None) -> str:
    """--task 未指定なら tasks.yaml の先頭タスク名を使う(構造可視化はタスク非依存)。"""
    if active_task is not None:
        return active_task
    import yaml

    tasks_path = _PROJECT_ROOT / "configs" / "components" / "tasks.yaml"
    with open(tasks_path, "r", encoding="utf-8") as fh:
        tasks = yaml.safe_load(fh)
    if not tasks:
        raise ValueError(f"{tasks_path} にタスク定義がありません。--task で明示してください。")
    return next(iter(tasks))


def build_and_visualize(config_path: str, output_dir: str | Path, active_task: str | None = None,
                        seed: int = 0) -> None:
    """YAML からネットワークを構築し、構造図一式を output_dir へ保存する。"""
    from src.core.config_manager import ConfigManager
    from src.core.NetworkBuilder import NetworkBuilder

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading config from {config_path} ...")
    manager = ConfigManager()
    config = manager.resolve(config_path, _default_task(active_task))

    # 構造可視化は GeNN のコンパイルも実行もしない (呼ぶのは _generate_global_matrices だけ)。
    # にもかかわらず NetworkBuilder.__init__ は pygenn.GeNNModel を作るため、backend 名が
    # pygenn の backend_modules に無いと **モデル生成の時点で KeyError** になる。CUDA 抜きで
    # ビルドされた pygenn では 'cuda' が登録されないので、`backend: cuda` の config を
    # そういう環境で可視化しようとすると落ちる (GPU が見えないだけなら落ちない。バックエンドの
    # 実体生成はビルド時まで遅延されるため、デバイスには触らない)。
    # ここでは何も走らせないので backend は結果に影響しない。この config は保存もしないため、
    # 「記録された backend を書き換える」問題も起きない。
    config.simulation.backend = "cpu"

    print("Building global network matrices (no GeNN compilation) ...")
    builder = NetworkBuilder(config)
    builder._generate_global_matrices()

    layout = builder.layout
    total = builder.total_neurons
    coords = builder.global_coords
    row, col, weights, delays = _coo_from_builder(builder)
    print(f"  {total} neurons, {row.size} synapses")

    has_space = coords is not None and np.all(np.isfinite(np.asarray(coords, dtype=np.float64)))

    # --- 図の生成 ---
    plot_connection_mask_coarse(
        row, col, layout, total, output_dir / "connection_mask_coarse.png",
    )
    if row.size:
        plot_delay_distribution(
            delays, row, col, layout, total, output_dir / "delay_distribution.png",
        )
        plot_weight_distributions(
            [0.0], [weights], output_dir / "weight_distribution.png",
            row=row, col=col, layout=layout, total_neurons=total,
        )

    if has_space:
        # 空間ネットワーク図は plotting.network.network に一本化 (サンプリング + 矢印/重み太さ)。
        # 描画自体はニューロン/エッジをサンプリングするため大規模でも軽いが、疎経路では
        # 密な重み行列 (N×N) を復元する必要があり、そのメモリだけが律速になる。
        DENSE_MAX_NEURONS = 20000  # N×N float64 復元の上限 (~3.2GB)
        if builder.global_weights is not None or total <= DENSE_MAX_NEURONS:
            network(
                weights=_dense_weights(builder), coords=coords, config=config,
                layout=layout, title="network_sample", save_path=str(output_dir),
                seed=seed,
            )
        else:
            print(f"  ニューロン数 {total} > {DENSE_MAX_NEURONS} のため "
                  f"空間ネットワーク図 (密行列復元) は省略しました。")
        if row.size:
            plot_empirical_connection_probability(
                coords, row, col, layout, output_dir / "connection_probability.png",
                connection_config=config.network.connection, seed=seed,
            )
    else:
        print("  座標が無い (no_space) ため空間依存の図はスキップしました。")

    print(f"Figures saved to: {output_dir}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="YAML 設定からネットワークを構築し、構造図一式を出力する。",
    )
    parser.add_argument("config", help="メイン設定 YAML へのパス (例: configs/test.yaml)")
    parser.add_argument(
        "-o", "--output", default="output",
        help="出力先ディレクトリ (デフォルト: output/)",
    )
    parser.add_argument(
        "--task", default=None,
        help="tasks.yaml 内のタスク名 (省略時は先頭タスク。構造可視化はタスク非依存)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="サンプリング図の乱数シード (デフォルト: 0)",
    )
    args = parser.parse_args(argv)

    # ConfigManager が configs/ を相対パスで参照するため、プロジェクトルートを基準にする。
    os.chdir(_PROJECT_ROOT)

    _register_plugins()
    build_and_visualize(args.config, args.output, active_task=args.task, seed=args.seed)


if __name__ == "__main__":
    main()
