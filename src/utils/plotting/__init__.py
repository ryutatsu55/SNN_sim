"""汎用の描画関数。

ここに置くのは「どの実験でも同じ意味を持つ図」だけ。ファイルの読み書きはせず
(保存する png を除く)、渡された配列と `NetworkLayout` だけで描く。

- 実験固有の見せ方 … `src/utils/experiments/`
- 数値計算         … `src/utils/analysis/`
"""
from .animation import spike_animation
from .area import draw_area, plot_area
from .distributions import (
    plot_avalanche_distribution,
    plot_delay_distribution,
    plot_distance_distribution,
    plot_synapse_value_distribution,
    plot_weight_distributions,
)
from .matrices import (
    plot_connection_mask_coarse,
    plot_empirical_connection_probability,
    plot_single_weight_matrix,
    plot_weight_panel,
)
from .network import axon_network, network
from .ordering import DEFAULT_ORDER_AXES, resolve_ordering
from .raster import plot_raster
from .traces import PQN_test, neuron_test, neuron_trace, stdp_window

__all__ = [
    "DEFAULT_ORDER_AXES",
    "PQN_test",
    "axon_network",
    "draw_area",
    "network",
    "neuron_test",
    "neuron_trace",
    "plot_area",
    "plot_avalanche_distribution",
    "plot_connection_mask_coarse",
    "plot_delay_distribution",
    "plot_distance_distribution",
    "plot_empirical_connection_probability",
    "plot_raster",
    "plot_single_weight_matrix",
    "plot_synapse_value_distribution",
    "plot_weight_distributions",
    "plot_weight_panel",
    "resolve_ordering",
    "spike_animation",
    "stdp_window",
]
