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
from contextlib import contextmanager


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_manager import ConfigManager  # noqa: E402
from src.core.layout import NetworkLayout  # noqa: E402
from src.core.output_manager import CONFIG_NAME  # noqa: E402
from src.utils.plotting.distributions import (  # noqa: E402
    delay_distribution,
    distance_distribution,
)
from src.utils.plotting.matrices import connection_mask, weight_matrix  # noqa: E402
from src.utils.plotting import ordering as ordering_module  # noqa: E402
from src.utils.plotting.ordering import block_ticks, resolve_ordering  # noqa: E402
from src.utils.plotting.raster import raster  # noqa: E402
from src.utils.runview import BuiltNetwork, Coo, MemoryWindow, Spikes  # noqa: E402

CONFIG_TEMPLATE = """
simulation: {{N: {total}, dt: 0.1, seed: 1}}
layout: {{assignment: {assignment}}}
inputs: {{GaussianNoise: {{enable: false}}}}
neurons:
  Exc: {{type: akita_escape_lif, mode: excitatory, polarity: excitatory, num: {num_exc}}}
  Inh: {{type: akita_escape_lif, mode: inhibitory, polarity: inhibitory, num: {num_inh}}}
synapses: {{}}
network:
  area: {{profile_name: no_space}}
  space: {{profile_name: no_space}}
  connection: {{profile_name: constant_prob_full, p: 1.0, allow_self_connections: false}}
  weight: {{profile_name: constant_zero}}
  delay: {{profile_name: constant}}
task: {{profile_name: test}}
meta: {{timestamp: test}}
"""


@contextmanager
def monkeypatched_axes(axes):
    """`available_order_axes()` が返す軸を差し替える。

    並べ替え軸は**図の引数ではなく layout から決まる** (`ordering.available_order_axes`)
    のが新しい約束なので、軸ごとの振る舞いを見るテストはそこを差し替える。

    各図のモジュールは名前で import しているので、**import 先を 1 つずつ**差し替える
    (`ordering` 側だけ書き換えても効かない)。

    `import src.utils.plotting.raster` ではモジュールを掴めない —— パッケージの
    `__init__` が同名の*関数* `raster` を re-export しているので属性参照がそちらに
    解決される。`importlib.import_module` なら確実にモジュール
    (`src/utils/CLAUDE.md` の「落とし穴」)。
    """
    import importlib

    targets = [ordering_module,
               importlib.import_module("src.utils.plotting.matrices"),
               importlib.import_module("src.utils.plotting.raster")]
    saved = [module.available_order_axes for module in targets]
    for module in targets:
        module.available_order_axes = lambda layout, *a, **k: axes
    try:
        yield
    finally:
        for module, original in zip(targets, saved):
            module.available_order_axes = original


def full_coo(total: int, *, weights=None, delays=None, coords=None) -> Coo:
    """全ペアの COO。密化は `matrices.py` が描画直前に自分で行う。"""
    row, col = np.meshgrid(np.arange(total), np.arange(total), indexing="ij")
    row, col = row.reshape(-1), col.reshape(-1)
    if weights is None:
        weights = np.random.default_rng(0).random(row.size)
    return Coo(row=row, col=col, weights=weights, delays=delays, shape=(total, total))


def built(layout, coo: Coo, *, coords=None) -> BuiltNetwork:
    """契約 (`src/utils/runview.py`) を満たす最小の `Built`。

    図は view からしかデータを取らないので、テストも view を組み立てて渡す。
    """
    from types import SimpleNamespace
    config = SimpleNamespace(
        simulation=SimpleNamespace(N=layout.total_neurons),
        network=SimpleNamespace(area=SimpleNamespace(profile_name="no_space"),
                                space=SimpleNamespace(),
                                connection=SimpleNamespace(profile_name="constant_prob_full")))
    return BuiltNetwork(run_dir=Path("."), config=config, layout=layout, coo=coo,
                        coords=coords)


def window(layout, times, ids) -> MemoryWindow:
    """ラスター用の最小の `Window`。"""
    from types import SimpleNamespace
    config = SimpleNamespace(simulation=SimpleNamespace(N=layout.total_neurons),
                             task=SimpleNamespace(duration=1000.0))
    return MemoryWindow(Path("."), config, layout,
                        spikes=Spikes(times=np.asarray(times), ids=np.asarray(ids)))


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
    # 入力は COO (全ペア)。密化は matrices.py が描画直前に自分で行う。
    row, col = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    row, col = row.reshape(-1), col.reshape(-1)
    weights = np.random.default_rng(0).random(row.size)

    matrix_path = tmp_path / "matrix.png"
    with monkeypatched_axes(("layer", "polarity")):
        weight_matrix(built(layout, Coo(row=row, col=col, weights=weights, delays=None,
                                        shape=(10, 10))), matrix_path)
        assert matrix_path.stat().st_size > 0

        raster_path = tmp_path / "raster.png"
        raster(window(layout, [0.0, 100.0, 200.0], [0, 5, 9]), raster_path)
        assert raster_path.stat().st_size > 0


def test_polarity_boundary_is_not_drawn():
    """E/I の切り替わりは境界としては残るが、描画対象からは外れること。

    赤/青の色分けが既にその位置を示しているので線は冗長。境界自体を消してしまうと
    `block_ticks` など位置を使う側が壊れるので、落とすのは描画側だけ。
    """
    layout = make_layout(6, 4, assignment="random")
    ordering = resolve_ordering(layout, ("polarity",))

    assert ordering.boundaries == [(6, 0)]
    assert ordering.visible_boundaries() == []
    # skip を空にすれば元の境界がそのまま出る
    assert ordering.visible_boundaries(skip=()) == [(6, 0)]


def test_module_boundaries_survive_the_polarity_filter(tmp_path):
    """`("module", "polarity")` ではモジュール境界だけが線になること。"""
    layout = make_layout(6, 4, assignment="random")
    layout.add_axis("module", np.array(["M0", "M0", "M1", "M1", "M2"] * 2))
    ordering = resolve_ordering(layout, ("module", "polarity"))

    # 元の境界にはモジュール(level 0)とモジュール内 E/I(level 1)の両方が居る
    assert {level for _, level in ordering.boundaries} == {0, 1}

    visible = ordering.visible_boundaries()
    assert [level for _, level in visible] == [0, 0]
    assert [pos for pos, _ in visible] == ordering.positions(level=0)

    # 描画も通ること (モジュールでブロック化したラスター)。軸を選ぶのは図ではなく
    # `available_order_axes()` なので、layout が module を持っていれば自動でこうなる。
    raster_path = tmp_path / "raster_module.png"
    raster(window(layout, [0.0, 100.0, 200.0], [0, 5, 9]), raster_path)
    assert raster_path.stat().st_size > 0


def test_coarse_mask_groups_by_the_given_axis(tmp_path):
    """粗視化図が `order_axes` に従ってブロック化すること (E/I 固定ではない)。

    絵そのものは比べようがないので、**同じ結合を別の軸で並べたら別の画像になる**ことと、
    指定した軸のブロックが実際に連続した表示位置を占めることを見る。前者だけだと
    order_axes を無視していても偶然通りうるので、両方要る。
    """
    layout = make_layout(6, 4, assignment="random")
    layout.add_axis("module", np.array(["M0", "M0", "M1", "M1", "M2"] * 2))
    row, col = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
    row, col = row.reshape(-1), col.reshape(-1)

    view = built(layout, Coo(row=row, col=col, weights=np.ones(row.size), delays=None,
                             shape=(10, 10)))
    paths = {}
    for name, axes in (("polarity", ("polarity",)),
                       ("module", ("module",)),
                       ("nested", ("module", "polarity")),
                       ("none", None)):
        paths[name] = tmp_path / f"coarse_{name}.png"
        with monkeypatched_axes(axes):
            connection_mask(view, paths[name])
        assert paths[name].stat().st_size > 0

    # 軸が違えば並びが違うので、画像も違う
    blobs = {name: path.read_bytes() for name, path in paths.items()}
    assert len({blobs["polarity"], blobs["module"], blobs["none"]}) == 3

    # module 指定時、同じモジュールのニューロンは連続した表示位置に固まる
    ordering = resolve_ordering(layout, ("module",))
    modules = layout.labels("module")[ordering.order]
    assert list(modules) == sorted(modules)


def test_coarse_mask_colors_cells_by_ei_block():
    """粗視化図の色が E/I ブロックを表し、濃さが密度になっていること。

    E/I を色で示すからこそ E/I の境界線を省ける、という図の前提そのもの。
    """
    from src.utils.plotting.common import BLOCK_COLORS
    from src.utils.plotting.matrices import _block_colored_density
    from matplotlib.colors import to_rgb

    # セル 0 = 興奮性、セル 1 = 抑制性 の 2x2。
    cell_of_rank = np.array([0, 0, 1, 1])
    per_cell = np.array([2.0, 2.0])
    is_exc = np.array([True, True, False, False])
    density = np.array([[0.5, 1.0], [0.0, 0.25]])

    rgb, max_density = _block_colored_density(density, cell_of_rank, per_cell, is_exc)

    assert max_density == 1.0
    # 密度最大のセル (0, 1) は pre=E, post=I なので EI の色そのもの。
    np.testing.assert_allclose(rgb[0, 1], to_rgb(BLOCK_COLORS["EI"]), atol=1e-6)
    # 密度 0 のセル (1, 0) は白。
    np.testing.assert_allclose(rgb[1, 0], (1.0, 1.0, 1.0), atol=1e-6)
    # 中間のセルは白とブロック色の間 (EE / II の色相を保ったまま薄い)。
    for cell, name in (((0, 0), "EE"), ((1, 1), "II")):
        colour = np.asarray(to_rgb(BLOCK_COLORS[name]))
        alpha = density[cell]
        np.testing.assert_allclose(rgb[cell], 1.0 - alpha * (1.0 - colour), atol=1e-6)


def test_coarse_mask_survives_more_blocks_than_ticks(tmp_path):
    """ブロック数が MAX_BLOCK_TICKS を超えても目盛りを諦めるだけで落ちないこと。"""
    from src.utils.plotting.matrices import MAX_BLOCK_TICKS

    n = 2 * (MAX_BLOCK_TICKS + 4)
    layout = make_layout(n // 2, n // 2, assignment="sequential")
    layout.add_axis("module", np.array([f"M{i}" for i in range(n)]))
    row = col = np.arange(n)

    out = tmp_path / "many_blocks.png"
    view = built(layout, Coo(row=row, col=col, weights=np.ones(row.size), delays=None,
                             shape=(n, n)))
    with monkeypatched_axes(("module",)):
        connection_mask(view, out)
    assert out.stat().st_size > 0


def test_missing_axis_is_reported():
    layout = make_layout(3, 2)
    with pytest.raises(Exception):
        resolve_ordering(layout, ("layer",))


def test_delay_and_distance_share_one_histogram(tmp_path):
    """遅延版と距離版は `_synapse_value_distribution` の薄い包み。

    骨格が 1 つであることの担保。どちらも同じ view から値を取り、軸ラベルだけが違う。
    """
    layout = make_layout(6, 4, assignment="sequential")
    coords = np.random.default_rng(0).random((10, 3)) * 100.0
    row = np.array([0, 1, 2, 6, 7])
    col = np.array([1, 2, 7, 0, 8])
    view = built(layout,
                 Coo(row=row, col=col, weights=np.ones(row.size),
                     delays=np.linspace(1.0, 5.0, row.size), shape=(10, 10)),
                 coords=coords)

    distance_path = tmp_path / "distance.png"
    distance_distribution(view, distance_path)
    assert distance_path.stat().st_size > 0

    delay_path = tmp_path / "delay.png"
    delay_distribution(view, delay_path)
    assert delay_path.stat().st_size > 0


def test_synapse_value_distribution_rejects_no_synapses(tmp_path):
    """結合ゼロは `MissingData`。**空配列の mean/max を呼ばない。**

    「描けない」を例外で言うのが契約の形。図の中で分岐して空の図を出すと、
    「結合が無かった」のか「描画に失敗した」のかが後から区別できない。
    """
    from src.utils.runview import MissingData

    layout = make_layout(6, 4, assignment="sequential")
    empty = np.array([], dtype=np.int64)
    view = built(layout, Coo(row=empty, col=empty, weights=np.array([]),
                             delays=np.array([]), shape=(10, 10)))

    with pytest.raises(MissingData):
        delay_distribution(view, tmp_path / "empty.png")
