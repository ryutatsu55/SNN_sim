"""汎用の描画。

ここに置くのは「どの実験でも同じ意味を持つ図」だけ。ファイルの読み書きはせず
(保存する png と mp4 を除く)、渡された view から必要なものを取って描く。

**全関数が `(view, out_path)` を取る** —— `view` は `src/utils/runview.py` の契約
(`Built` / `Window` / `Series`)。おかげで呼び出し側は「どの run か」と「どこへ出すか」
しか渡さない。持っていないデータ (`no_space` の座標、トレースを採っていない run の V/I)
を要求すると view が `MissingData` を投げるので、**図の側に「持っていなければ飛ばす」の
分岐は書かない**。それを捕まえるのは出力の登録簿 (`scripts/<実験>/report/`)。

例外は 2 つ:

- `ax` を取るプリミティブ (`draw_area` / `draw_discrete_distribution` /
  `draw_block_boundaries`)。図の生成と保存は呼び出し側の責任で、複数の図が同じ部品を
  使い回すためにある。
- `model_test.py`。モデル単体を手で駆動した結果を描く図で、run ではないので契約を
  取らない (理由はそのモジュールの docstring)。

**実験固有の図はここに置かない。** 実験は自分の `scripts/<実験>/figures/` に持つ
(必要なら複製する)。ここに集まるのは、実験に依存しない道具 (`scripts/tools/`) が使う図。
判断基準は `src/utils/CLAUDE.md`。

- 数値計算 … `src/utils/analysis/`
- 読み出し契約 … `src/utils/runview.py`
"""
from .animation import spike_animation
from .area import area_figure, draw_area
from .distributions import (
    avalanche_distribution,
    delay_distribution,
    distance_distribution,
    draw_discrete_distribution,
    weight_distribution,
)
from .matrices import (
    connection_mask,
    densify,
    empirical_connection_probability,
    weight_matrix,
    weight_panel,
)
from .model_test import PQN_test, neuron_test, stdp_window
from .network import axon_network, network
from .ordering import (
    DEFAULT_ORDER_AXES,
    GROUPED_ORDER_AXES,
    available_order_axes,
    draw_block_boundaries,
    resolve_ordering,
)
from .raster import raster
from .traces import neuron_trace

__all__ = [
    "DEFAULT_ORDER_AXES",
    "GROUPED_ORDER_AXES",
    "PQN_test",
    "area_figure",
    "available_order_axes",
    "avalanche_distribution",
    "axon_network",
    "connection_mask",
    "delay_distribution",
    "densify",
    "distance_distribution",
    "draw_area",
    "draw_block_boundaries",
    "draw_discrete_distribution",
    "empirical_connection_probability",
    "network",
    "neuron_test",
    "neuron_trace",
    "raster",
    "resolve_ordering",
    "spike_animation",
    "stdp_window",
    "weight_distribution",
    "weight_matrix",
    "weight_panel",
]
