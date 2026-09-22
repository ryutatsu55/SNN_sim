"""YAML 設定からネットワークを構築し、構造図一式を出力する。

    python -m scripts.tools.visualize_network_structure configs/test.yaml -o output/

入力は **run ディレクトリではなく config** で、NetworkBuilder を実際に走らせる
(GeNN のコンパイルと実行は行わず、呼ぶのは `_generate_global_matrices()` だけ)。

**特定の実験の持ち物ではないので `scripts/tools/` に置く。** 描く内容はどれも
「この config が作ったネットワークはどんな形か」であって、実験の都合では変わらない。
描画は共有層 (`src/utils/plotting/`) をそのまま使う。

ただし実験が**自分の構造図を持ちたくなったら、ここを共有せず複製する**こと
(`scripts/develop/report/structure.py` がその例)。何をどんな体裁で出すかは実験ごとに
細部が違ってくるので、汎用化にこだわって引数を肥大化させない。

ビルドした結果は `BuiltNetwork` (`src/utils/runview.py`) に載せてから図へ渡す。
図が `NetworkBuilder` の内部 API を直に叩かないのはそのため —— 図の側から見ると、
config からビルドしたものも run を再ビルドしたものも同じ `Built` でしかない。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config_manager import load_yaml
from src.utils.analysis.connectivity import (
    bridge_hop_matrix,
    format_group_connection_probability,
    format_hop_connection_probability,
    group_connection_probability,
    hop_connection_probability,
)
from src.utils.plotting import (
    area_figure,
    available_order_axes,
    axon_network,
    connection_mask,
    delay_distribution,
    distance_distribution,
    empirical_connection_probability,
    network,
    weight_distribution,
)
from src.utils.runview import BuiltNetwork, guard, optional

# 出す図。**1 行 1 出力。** 足すならここへ 1 行足す
# (`scripts/develop/report/structure.py` と同じ形)。
FIGURES = (
    ("area", area_figure, "area.png"),
    ("connection mask", connection_mask, "connection_mask_coarse.png"),
    ("delay distribution", delay_distribution, "delay_distribution.png"),
    ("weight distribution", weight_distribution, "weight_distribution.png"),
    ("network sample", network, "network_sample.png"),
    ("axon network", axon_network, "axon_network.png"),
    # 「その距離のペアのうち何割が繋がったか」(確率) と「実際に張られた結合の長さが
    # 何本ずつか」(件数) の 2 枚。分母が違うので、片方が単調減少でももう片方はピークを持つ。
    ("connection probability", empirical_connection_probability,
     "connection_probability.png"),
    ("distance distribution", distance_distribution, "distance_distribution.png"),
)


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
    tasks_path = _PROJECT_ROOT / "configs" / "components" / "tasks.yaml"
    tasks = load_yaml(tasks_path)
    if not tasks:
        raise ValueError(f"{tasks_path} にタスク定義がありません。--task で明示してください。")
    return next(iter(tasks))


def report_connection_probability(built, out_dir: Path) -> None:
    """層内 / 層間の結合確率を標準出力へ表で出す。

    軸は粗視化図と同じ最外軸 (`available_order_axes()[0]`) を使う。図で見えている
    濃淡が何倍の差なのかは絵からは読めないので、そこだけ数字で補う。

    エリアがモジュールとブリッジに分かれている (= `allow_soma: false` の part を持つ)
    なら、**経由するブリッジの本数**でグループ対をまとめ直した表も続けて出す。
    K×K の表は同一モジュール (対角) と「それ以外」(非対角) の 2 つにしか割れないが、
    こちらは非対角を 1 ホップ / 2 ホップ / … へ分解する。
    """
    axes = available_order_axes(built.layout)
    if not axes:
        return
    wiring = built.wiring()
    result = group_connection_probability(wiring.row, wiring.col, built.layout, axes[0])
    print()
    print(format_group_connection_probability(result))
    print()

    area = optional(built.area)
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


def emit(built, output_dir: str | Path) -> None:
    """構造図一式を `output_dir` へ保存する。

    **1 枚描けなくても止めない。** `no_space` の config に座標が無く、`constant_prob` は
    軸索の幾何を持たないが、それは異常ではないので `MissingData` として 1 枚ずつ
    報告される (`guard`)。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {built.total_neurons} neurons, {built.wiring().num_synapses} synapses")

    for label, draw, name in FIGURES:
        guard(label, draw, built, output_dir / name)

    report_connection_probability(built, output_dir)
    print(f"Figures saved to: {output_dir}")


def build_and_visualize(config_path: str, output_dir: str | Path,
                        active_task: str | None = None) -> None:
    """YAML からネットワークを構築し、構造図一式を output_dir へ保存する。"""
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

    emit(BuiltNetwork.from_builder(builder, output_dir), output_dir)


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
    args = parser.parse_args(argv)

    # ConfigManager が configs/ を相対パスで参照するため、プロジェクトルートを基準にする。
    os.chdir(_PROJECT_ROOT)

    _register_plugins()
    build_and_visualize(args.config, args.output, active_task=args.task)


if __name__ == "__main__":
    main()
