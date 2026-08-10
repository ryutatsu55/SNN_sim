"""表示用の並べ替え軸 (src/utils/plotting/ordering.py) の検証。

軸が 1 つのとき (既定の polarity) は従来の「興奮性を先頭ブロックに置く」動作そのもの、
軸を重ねたときは外側の軸でブロック化し、内側の軸でその中を並べ替えることを確認する。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_manager import ConfigManager  # noqa: E402
from src.core.layout import NetworkLayout  # noqa: E402
from src.core.output_manager import CONFIG_NAME  # noqa: E402
from src.utils.plotting.matrices import plot_single_weight_matrix  # noqa: E402
from src.utils.plotting.ordering import block_ticks, resolve_ordering  # noqa: E402
from src.utils.plotting.raster import plot_raster  # noqa: E402

CONFIG_TEMPLATE = """
simulation: {{N: {total}, dt: 0.1, seed: 1}}
layout: {{assignment: {assignment}}}
inputs: {{GaussianNoise: {{enable: false}}}}
neurons:
  Exc: {{type: akita_escape_lif, mode: excitatory, polarity: excitatory, num: {num_exc}}}
  Inh: {{type: akita_escape_lif, mode: inhibitory, polarity: inhibitory, num: {num_inh}}}
synapses: {{}}
network:
  space: {{profile_name: no_space}}
  connection: {{profile_name: constant_prob_full, p: 1.0, allow_self_connections: false}}
  weight: {{profile_name: constant_zero}}
  delay: {{profile_name: constant}}
task: {{profile_name: test}}
meta: {{timestamp: test}}
"""


def make_layout(num_exc: int, num_inh: int, assignment: str = "sequential") -> NetworkLayout:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / CONFIG_NAME
        path.write_text(
            CONFIG_TEMPLATE.format(total=num_exc + num_inh, num_exc=num_exc,
                                   num_inh=num_inh, assignment=assignment),
            encoding="utf-8",
        )
        return NetworkLayout.from_config(ConfigManager().load_resolved(path))


def test_polarity_ordering_puts_excitatory_first():
    layout = make_layout(6, 4, assignment="random")
    ordering = resolve_ordering(layout, ("polarity",))

    polarity = layout.labels("polarity")[ordering.order]
    assert list(polarity[:6]) == ["excitatory"] * 6
    assert list(polarity[6:]) == ["inhibitory"] * 4
    # 境界は E/I の 1 本だけ、最外レベル。
    assert ordering.boundaries == [(6, 0)]
    assert block_ticks(ordering, 10) == [0, 5, 9]


def test_ordering_disabled_without_axes_or_layout():
    layout = make_layout(3, 2)
    for ordering in (resolve_ordering(layout, None),
                     resolve_ordering(layout, ()),
                     resolve_ordering(None, ("polarity",))):
        assert not ordering.enabled
        assert ordering.boundaries == []
        matrix = np.arange(25).reshape(5, 5)
        np.testing.assert_array_equal(ordering.apply(matrix), matrix)


def test_rank_is_the_inverse_of_order():
    layout = make_layout(7, 3, assignment="random")
    ordering = resolve_ordering(layout, ("polarity",))
    np.testing.assert_array_equal(ordering.rank[ordering.order], np.arange(10))


def test_nested_axes_block_by_outer_then_sort_inner():
    layout = make_layout(6, 4, assignment="random")
    # 外部軸を後から差し込む (通常はコンポーネントの describe_axes() が供給する)。
    layer = np.array(["L1", "L1", "L2", "L2", "L3"] * 2)
    layout.add_axis("layer", layer)

    ordering = resolve_ordering(layout, ("layer", "polarity"))
    layers = layout.labels("layer")[ordering.order]
    polarity = layout.labels("polarity")[ordering.order]

    # 層は連続したブロックになり、各層の中では興奮性が先に来る。
    assert list(layers) == sorted(layers)
    for name in ("L1", "L2", "L3"):
        inside = polarity[layers == name]
        assert list(inside) == sorted(inside)

    # 層が切り替わる位置はレベル 0、層内で E→I が切り替わる位置はレベル 1。
    levels = {level for _, level in ordering.boundaries}
    assert levels == {0, 1}
    outer = ordering.positions(level=0)
    assert outer == [int(np.count_nonzero(layers == "L1")),
                     int(np.count_nonzero(layers != "L3"))]


def test_plots_accept_nested_axes(tmp_path):
    layout = make_layout(6, 4, assignment="random")
    layout.add_axis("layer", np.array(["L1", "L1", "L2", "L2", "L3"] * 2))
    weights = np.random.default_rng(0).random((10, 10))

    matrix_path = tmp_path / "matrix.png"
    plot_single_weight_matrix(weights, layout, matrix_path, "W",
                              order_axes=("layer", "polarity"))
    assert matrix_path.stat().st_size > 0

    raster_path = tmp_path / "raster.png"
    plot_raster(np.array([0.0, 100.0, 200.0]), np.array([0, 5, 9]), raster_path, "R",
                layout=layout, order_axes=("layer", "polarity"))
    assert raster_path.stat().st_size > 0


def test_missing_axis_is_reported():
    layout = make_layout(3, 2)
    with pytest.raises(Exception):
        resolve_ordering(layout, ("layer",))
