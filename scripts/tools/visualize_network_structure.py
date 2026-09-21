"""YAML 設定からネットワークを構築し、構造図一式を出力する。

入力は **run ディレクトリではなく config** で、NetworkBuilder を実際に走らせる
(GeNN のコンパイルと実行は行わず、呼ぶのは `_generate_global_matrices()` だけ)。

    python scripts/tools/visualize_network_structure.py configs/test.yaml -o output/

**特定の実験の持ち物ではないので `scripts/tools/` に置く。** 描く内容はどれも
「この config が作ったネットワークはどんな形か」であって、実験の都合では変わらない。

ただし実験が**自分の構造図を持ちたくなったら、ここを共有せず複製する**こと
(`scripts/develop/structure.py` がその例)。空間構造を持たせるか、どんな図をどんな形式で
出すかは実験ごとに細部が違ってくるので、汎用化にこだわって引数を肥大化させない。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# プロジェクトルートにパスを通す (scripts/tools/ から 2 階層上)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.analysis.connectivity import (
    bridge_hop_matrix,
    format_group_connection_probability,
    format_hop_connection_probability,
    group_connection_probability,
    hop_connection_probability,
)
from src.utils.plotting.area import plot_area
from src.utils.plotting.distributions import (
    plot_delay_distribution,
    plot_distance_distribution,
    plot_weight_distributions,
)
from src.utils.plotting.matrices import (
    plot_connection_mask_coarse,
    plot_empirical_connection_probability,
)
from src.utils.plotting.network import axon_network, network
from src.utils.plotting.ordering import DEFAULT_ORDER_AXES

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _register_plugins():
    """@register デコレータを走らせてコンポーネントをレジストリに登録する。"""
    import src.models.neurons.pqn_float  # noqa: F401
    import src.models.neurons.pqn_int  # noqa: F401
    import src.models.neurons.akita_escape_lif  # noqa: F401
    import src.models.neurons.akita_escape_lif_physical  # noqa: F401
    import src.models.neurons.lif  # noqa: F401
    import src.models.network.area  # noqa: F401
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


def _default_order_axes(layout) -> tuple[str, ...]:
    """`--order-axes` 未指定時の並べ替え軸。

    `module` 軸がある run (複合エリア + `space: area_uniform`) では**モジュールで
    ブロック化し、その中で E→I** にする。粗視化図はモジュール間結合の粗密を見るための
    図なので、E/I だけで切るとブロック構造が見えない。`module` を持たない run では
    従来どおり `("polarity",)`。
    """
    if layout is not None and layout.has_axis("module"):
        return ("module", "polarity")
    return DEFAULT_ORDER_AXES


def _report_connection_probability(coo, layout, axes: tuple[str, ...], area=None) -> None:
    """層内 / 層間の結合確率を標準出力へ表で出す。

    軸は粗視化図と同じ最外軸 (`axes[0]`) を使う。並べ替えなし (`--order-axes none`) や
    軸を持たない run では出しようがないので黙って飛ばす。

    エリアがモジュールとブリッジに分かれている (= `allow_soma: false` の part を持つ)
    なら、**経由するブリッジの本数**でグループ対をまとめ直した表も続けて出す。
    K×K の表は同一モジュール (対角) と「それ以外」(非対角) の 2 つにしか割れないが、
    こちらは非対角を 1 ホップ / 2 ホップ / … へ分解する。
    """
    if not axes:
        return
    axis = axes[0]
    result = group_connection_probability(coo.row, coo.col, layout, axis)
    print()
    print(format_group_connection_probability(result))
    print()
    _report_bridge_hops(result, area)


def _report_bridge_hops(result, area) -> None:
    """ホップ数別の結合確率。エリアがブリッジを持たない構成なら理由を 1 行出して飛ばす。"""
    if area is None:
        return
    try:
        hops = bridge_hop_matrix(area, result.names)
    except ValueError as exc:
        print(f"  ブリッジのホップ数別の集計はスキップしました: {exc}")
        print()
        return
    print(format_hop_connection_probability(hop_connection_probability(result, hops)))
    print()


def build_and_visualize(config_path: str, output_dir: str | Path, active_task: str | None = None,
                        seed: int = 0, order_axes: tuple[str, ...] | None = None) -> None:
    """YAML からネットワークを構築し、構造図一式を output_dir へ保存する。

    Args:
        order_axes: 粗視化図の並べ替え軸。None なら `_default_order_axes()` が run を見て決める。
    """
    from src.core.config_manager import ConfigManager
    from src.core.NetworkBuilder import NetworkBuilder

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

    visualize_structure(builder, config, output_dir, seed=seed, order_axes=order_axes)


def visualize_structure(builder, config, output_dir: str | Path, seed: int = 0,
                        order_axes: tuple[str, ...] | None = None,
                        geometry=None) -> None:
    """**ビルド済みの** NetworkBuilder から構造図一式を output_dir へ保存する。

    呼び出し経路は 2 つある。`build_and_visualize()` が config からビルドして呼ぶ経路と、
    シミュレーションを回すスクリプトが自分の builder を渡して呼ぶ経路。後者では
    **ここで作り直してはならない** — config の seed が未指定なら `resolve()` のたびに
    別の seed が引かれ、図が「実際に走らせたネットワーク」と別物になる。

    Args:
        builder: `build()` か `_generate_global_matrices()` を通した後の NetworkBuilder
        config: その builder が使った解決済み config
        order_axes: 粗視化図の並べ替え軸。None なら `_default_order_axes()` が run を見て決める。
        geometry: axon_network.png に使う軸索幾何。None なら builder のコネクタから取る。
            **損傷実験では必ず渡すこと** — コネクタが持つ幾何は `replace_global_coo()` の
            影響を受けないので、既定のままだと切断したはずの結合まで描かれる
            (`analysis/axons.subset_geometry()` で絞ったものを渡す)。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layout = builder.layout
    total = builder.total_neurons
    coords = builder.global_coords
    # 疎/密の違いは builder が吸収済み。ここから先は COO しか見ない。
    coo = builder.global_coo()
    print(f"  {total} neurons, {coo.num_synapses} synapses")

    has_space = coords is not None and np.all(np.isfinite(np.asarray(coords, dtype=np.float64)))

    # --- 図の生成 ---
    # エリアの図は has_space ではなく area.is_bounded で判定する。両者は独立で、
    # 有界なエリアは no_space の座標とも共存しうる (逆もまた然り)。
    area = builder.area
    if plot_area(area, output_dir / "area.png",
                 title=f"Area: {config.network.area.profile_name}"):
        print(f"  Area visualization saved to {output_dir}/area.png")
    else:
        print(f"  エリアが無界 ({config.network.area.profile_name}) のため "
              f"area.png はスキップしました。")

    axes = _default_order_axes(layout) if order_axes is None else order_axes
    plot_connection_mask_coarse(
        coo.row, coo.col, layout, total, output_dir / "connection_mask_coarse.png",
        order_axes=axes,
    )
    print(f"  Connection mask grouped by {' > '.join(axes) if axes else 'global ID order'}")

    # 粗視化図と**同じ軸**で層内/層間の結合確率を数値でも出す。図で見えている濃淡が
    # 何倍の差なのかは絵からは読めないので、そこだけ数字で補う。
    _report_connection_probability(coo, layout, axes, area=area)
    if coo.num_synapses:
        plot_delay_distribution(
            coo.delays, coo.row, coo.col, layout, total,
            output_dir / "delay_distribution.png",
        )
        plot_weight_distributions(
            [0.0], [coo.weights], output_dir / "weight_distribution.png",
            row=coo.row, col=coo.col, layout=layout, total_neurons=total,
        )

    if has_space:
        # 空間ネットワーク図は plotting.network.network に一本化 (サンプリング + 矢印/重み太さ)。
        # COO をそのまま渡すので、密行列の復元は要らず N の上限も無い。
        network(
            coo.row, coo.col, coo.weights, coords, config,
            output_dir / "network_sample.png",
            layout=layout, title="network_sample", seed=seed, area=area,
        )

        # 軸索の折れ線で結合を描いた図。幾何を残すのは axon_growth だけなので、
        # 持っていないコネクタ (constant_prob など) は素通りさせる。
        if geometry is None:
            geometry = getattr(builder.connection, "axon_geometry", lambda: None)()
        if geometry is not None:
            axon_network(
                geometry, coords, config, output_dir / "axon_network.png",
                layout=layout, title="axon_network", seed=seed, area=area,
            )
        else:
            print(f"  connection ({config.network.connection.profile_name}) が軸索の幾何を"
                  f"持たないため axon_network.png はスキップしました。")

        if coo.num_synapses:
            # 「その距離のペアのうち何割が繋がったか」(確率) と
            # 「実際に張られた結合の長さが何本ずつか」(件数) の 2 枚。分母が違うので
            # 片方が単調減少でももう片方はピークを持つ。
            plot_empirical_connection_probability(
                coords, coo.row, coo.col, layout,
                output_dir / "connection_probability.png",
                connection_config=config.network.connection, seed=seed,
            )
            plot_distance_distribution(
                coords, coo.row, coo.col, layout, total,
                output_dir / "distance_distribution.png",
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
    parser.add_argument(
        "--order-axes", nargs="+", default=None, metavar="AXIS",
        help="connection_mask_coarse.png を並べ替える軸を外側から順に。"
             " 省略時は module 軸があれば 'module polarity'、無ければ 'polarity'。"
             " 例: --order-axes module。'none' でグローバル ID の順のまま",
    )
    args = parser.parse_args(argv)

    order_axes = args.order_axes
    if order_axes is not None:
        # 'none' 1 語で「並べ替えない」。weight_track.py の --order-axes と同じ約束。
        order_axes = () if [a.lower() for a in order_axes] == ["none"] else tuple(order_axes)

    # ConfigManager が configs/ を相対パスで参照するため、プロジェクトルートを基準にする。
    os.chdir(_PROJECT_ROOT)

    _register_plugins()
    build_and_visualize(args.config, args.output, active_task=args.task, seed=args.seed,
                        order_axes=order_axes)


if __name__ == "__main__":
    main()
