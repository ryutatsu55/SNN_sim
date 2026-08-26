"""群間結合確率 (src/utils/analysis/connectivity.py) の検証。

守りたいのは 4 つ:
  1. 分母が「ありうるペア数」であること — グループの大きさが違っても比較できる根拠
  2. 対角の分母から自己ペアが抜けていること (allow_self_connections=False が普通なので、
     含めると到達しえないペアで割ることになり層内確率が過小に出る)
  3. 方向つきであること (p[i,j] と p[j,i] は別物)
  4. 層内/層間のプールが「確率の単純平均」ではなく「分子と分母をそれぞれ足して割る」こと
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.layout import NetworkLayout  # noqa: E402
from src.utils.analysis.connectivity import (  # noqa: E402
    bridge_hop_matrix,
    format_group_connection_probability,
    format_hop_connection_probability,
    group_connection_probability,
    hop_connection_probability,
)


def make_layout(sizes: dict[str, int], axis: str = "module") -> NetworkLayout:
    """{グループ名: 人数} から、その軸だけを持つ layout を作る。"""
    labels = np.array([name for name, n in sizes.items() for _ in range(n)])
    layout = NetworkLayout(["P"] * labels.size)
    layout.add_axis(axis, labels)
    return layout


def all_pairs(ids) -> tuple[list[int], list[int]]:
    """自己結合を除く全ペア (i -> j)。"""
    row, col = [], []
    for a in ids:
        for b in ids:
            if a != b:
                row.append(a)
                col.append(b)
    return row, col


def test_within_is_one_when_each_group_is_fully_connected():
    """層内が全結合なら層内確率はちょうど 1.0。

    自己ペアが分母から抜けていないと 3/4 = 0.75 になるので、ここが 2 の検査点。
    """
    layout = make_layout({"M0": 4, "M1": 4, "M2": 4})
    row, col = [], []
    for g in range(3):
        r, c = all_pairs(range(4 * g, 4 * g + 4))
        row += r
        col += c

    result = group_connection_probability(np.array(row), np.array(col), layout, "module")
    np.testing.assert_allclose(np.diag(result.possible), [12, 12, 12])   # 4*3, 自己ペア除外
    np.testing.assert_allclose(result.within, [1.0, 1.0, 1.0])
    assert result.within_probability == pytest.approx(1.0)
    assert result.between_probability == pytest.approx(0.0)
    assert result.segregation == np.inf          # 層間 0 = 完全に分離


def test_include_self_pairs_only_changes_the_diagonal_denominator():
    layout = make_layout({"M0": 4, "M1": 4})
    row, col = all_pairs(range(4))

    excl = group_connection_probability(np.array(row), np.array(col), layout, "module")
    incl = group_connection_probability(np.array(row), np.array(col), layout, "module",
                                        include_self_pairs=True)

    np.testing.assert_array_equal(excl.counts, incl.counts)
    np.testing.assert_allclose(np.diag(incl.possible), [16, 16])
    np.testing.assert_allclose(np.diag(excl.possible), [12, 12])
    # 非対角は自己ペアと無関係なので不変
    np.testing.assert_array_equal(excl.possible[0, 1], incl.possible[0, 1])
    assert excl.within[0] == pytest.approx(1.0)
    assert incl.within[0] == pytest.approx(12 / 16)


def test_probability_is_directed():
    """row=送信 / col=受信 なので、片方向だけの結合は非対称に出る。"""
    layout = make_layout({"M0": 2, "M1": 5})
    # M0 -> M1 だけ 3 本、逆向きは 0 本
    result = group_connection_probability(
        np.array([0, 0, 1]), np.array([2, 3, 4]), layout, "module")

    assert result.probability[0, 1] == pytest.approx(3 / (2 * 5))
    assert result.probability[1, 0] == pytest.approx(0.0)
    np.testing.assert_array_equal(result.sizes, [2, 5])


def test_pooled_within_is_not_the_mean_of_per_group_probabilities():
    """大きさの違うグループがあると、プール値と単純平均はずれる。

    プール = Σ対角counts / Σ対角possible が欲しい量 (「層内のペアを 1 つ引いたら
    結合している確率」)。単純平均だと小さいグループが過剰に効く。
    """
    layout = make_layout({"M0": 2, "M1": 10})
    # M0 は全結合 (2 本 / 2 ペア = 1.0)、M1 は 9 本 / 90 ペア = 0.1
    row, col = all_pairs(range(2))
    row += [2 + i for i in range(9)]
    col += [3 + i for i in range(9)]

    result = group_connection_probability(np.array(row), np.array(col), layout, "module")
    np.testing.assert_allclose(result.within, [1.0, 0.1])

    pooled = (2 + 9) / (2 + 90)
    assert result.within_probability == pytest.approx(pooled)
    assert result.within_probability != pytest.approx(np.mean([1.0, 0.1]))


def test_empty_coo_gives_zero_probability_not_nan():
    """結合が 1 本も無くても「ペアはある」ので確率は 0。NaN は分母 0 のときだけ。"""
    layout = make_layout({"M0": 3, "M1": 3})
    result = group_connection_probability(np.array([], dtype=np.int64),
                                          np.array([], dtype=np.int64), layout, "module")
    np.testing.assert_allclose(result.probability, np.zeros((2, 2)))
    assert result.within_probability == pytest.approx(0.0)
    assert np.isnan(result.segregation)          # 0 / 0


def test_singleton_group_has_no_within_pairs():
    """1 人だけのグループは層内ペアが 0 なので確率が定義できない (NaN)。"""
    layout = make_layout({"M0": 1, "M1": 3})
    result = group_connection_probability(np.array([1]), np.array([2]), layout, "module")

    assert result.possible[0, 0] == 0
    assert np.isnan(result.probability[0, 0])
    # 表では "-" になり、数値として印字されない
    assert "-" in format_group_connection_probability(result)


def test_numeric_axis_is_grouped_in_numeric_order():
    """数値軸でも辞書順にならないこと ("10" < "9" のずれを踏まない)。"""
    layout = NetworkLayout(["P"] * 3)
    layout.add_axis("depth", np.array([9.0, 10.0, 9.0]))
    result = group_connection_probability(np.array([0]), np.array([1]), layout, "depth")

    assert result.names == ["9.0", "10.0"]
    np.testing.assert_array_equal(result.sizes, [2, 1])   # 9.0 が 2 人、10.0 が 1 人
    assert result.counts[0, 1] == 1


def test_format_reports_the_axis_and_both_summaries():
    layout = make_layout({"M0": 3, "M1": 3})
    result = group_connection_probability(np.array([0, 3]), np.array([1, 4]), layout, "module")
    text = format_group_connection_probability(result)

    assert "'module'" in text
    assert "within-module" in text
    assert "between-module" in text
    # 行数 = ヘッダ 2 + グループ 2 + 区切り 1 + サマリ 3
    assert len(text.splitlines()) == 8


# ---------------------------------------------------------------- ブリッジ経由のホップ数

def make_grid_area(**overrides):
    """`modular_grid` エリア (既定 2x2、ブリッジに soma は置かない) を作る。"""
    from types import SimpleNamespace

    from src.models.network.area import ModularGridArea

    params = dict(side=100.0, bridge_width=10.0, spacing=200.0, num_modules=4,
                  soma_in_bridge=False)
    params.update(overrides)
    return ModularGridArea(SimpleNamespace(**params))


def test_hop_matrix_is_the_module_graph_distance():
    """2x2 格子は「隣は 1 ホップ、対角は 2 ホップ」。斜めのブリッジは張らないため。

    モジュール番号は行優先 (M0 が左下、M1 が右下、M2 が左上、M3 が右上) なので、
    2 ホップになるのは M0-M3 と M1-M2 の 2 対だけ。
    """
    hops = bridge_hop_matrix(make_grid_area(), ["M0", "M1", "M2", "M3"])
    expected = np.array([[0, 1, 1, 2],
                         [1, 0, 2, 1],
                         [1, 2, 0, 1],
                         [2, 1, 1, 0]])
    assert np.array_equal(hops, expected)


def test_hop_matrix_counts_bridges_not_parts():
    """隣接モジュールは間にブリッジ part を 1 つ挟むが、ホップ数は 2 ではなく 1。

    数えるのは「通った part の数」ではなく「通ったブリッジの本数」なので、モジュール
    part はコスト 0。ここを取り違えると全部のホップ数が 2 倍になる。
    """
    hops = bridge_hop_matrix(make_grid_area(), ["M0", "M1"])
    assert hops[0, 1] == 1


def test_hierarchical_grid_has_one_hop_for_each_bridge():
    """階層格子は 16 モジュール / 20 ブリッジ。1 ホップの順序つき対はちょうど 2x20。

    最遠は対角クラスタの外側モジュール同士で 6 ホップ (外→内 2 + クラスタ間 1 +
    内→内 1 + クラスタ間 ... と経由する)。全対は 16x16 = 256 に閉じる。
    """
    from types import SimpleNamespace

    from src.models.network.area import HierarchicalModularGridArea

    area = HierarchicalModularGridArea(SimpleNamespace(
        side=100.0, bridge_width=10.0, spacing=200.0, soma_in_bridge=False))
    names = [f"C{c}-M{m}" for c in range(4) for m in range(4)]
    hops = bridge_hop_matrix(area, names)

    assert hops.shape == (16, 16)
    assert np.array_equal(hops, hops.T)          # 距離なので対称
    assert np.all(np.diag(hops) == 0)
    assert (hops == 1).sum() == 2 * area.n_bridges
    assert hops.max() == 6
    assert (hops >= 0).all()                     # 全体が 1 つの連結成分


def test_hop_zero_probability_equals_pooled_within():
    """ホップ 0 は同一モジュールなので、K×K 表の対角プールとぴったり一致する。"""
    layout = make_layout({"M0": 4, "M1": 4, "M2": 4, "M3": 4})
    row, col = [], []
    for g in range(4):
        r, c = all_pairs(range(4 * g, 4 * g + 4))
        row += r
        col += c
    row += [0, 1]        # M0 -> M1 (1 ホップ) と M0 -> M3 (2 ホップ)
    col += [4, 12]

    result = group_connection_probability(np.array(row), np.array(col), layout, "module")
    hops = bridge_hop_matrix(make_grid_area(), result.names)
    hop_result = hop_connection_probability(result, hops)

    assert np.array_equal(hop_result.levels, [0, 1, 2])
    assert hop_result.probability[0] == pytest.approx(result.within_probability)
    # 1 ホップの対は 8 組 (順序つき)、各 4x4 = 16 ペアで分母 128。実結合は 1 本。
    assert hop_result.group_pairs.tolist() == [4, 8, 4]
    assert hop_result.counts.tolist() == [48, 1, 1]
    assert hop_result.possible.tolist() == [48, 128, 64]


def test_disconnected_modules_are_reported_separately():
    """ブリッジで繋がっていないモジュール対は -1 (到達不能) になり、末尾の段に集まる。"""
    from types import SimpleNamespace

    from src.models.network.area import CompositeArea

    area = CompositeArea(SimpleNamespace(op="union", parts=[
        {"type": "rect", "x_range": [0.0, 100.0], "y_range": [0.0, 100.0], "name": "M0"},
        {"type": "rect", "x_range": [95.0, 200.0], "y_range": [45.0, 55.0], "name": "B0-1",
         "allow_soma": False},
        {"type": "rect", "x_range": [195.0, 295.0], "y_range": [0.0, 100.0], "name": "M1"},
        {"type": "rect", "x_range": [500.0, 600.0], "y_range": [0.0, 100.0], "name": "M2"},
    ]))
    hops = bridge_hop_matrix(area, ["M0", "M1", "M2"])
    assert hops[0, 1] == 1
    assert hops[0, 2] == -1 and hops[1, 2] == -1

    layout = make_layout({"M0": 2, "M1": 2, "M2": 2})
    result = group_connection_probability(np.array([0]), np.array([2]), layout, "module")
    hop_result = hop_connection_probability(result, hops)
    assert hop_result.levels.tolist() == [0, 1, -1]        # -1 は末尾
    assert hop_result.counts.tolist() == [0, 1, 0]


def test_bridge_hops_need_a_part_without_somas():
    """全 part に soma を置ける構成 (soma_in_bridge: true) では定義できないので ValueError。

    黙って 0 を返すと「ブリッジを通らずに繋がっている」という嘘の表が出る。
    """
    with pytest.raises(ValueError, match="allow_soma"):
        bridge_hop_matrix(make_grid_area(soma_in_bridge=True), ["M0", "M1"])


def test_hop_format_lists_every_level_including_unreachable():
    """表にはホップ数の段が全部出る。到達不能は数値ではなく `n/a` で示す。"""
    layout = make_layout({"M0": 2, "M1": 2})
    result = group_connection_probability(np.array([0]), np.array([1]), layout, "module")
    hops = np.array([[0, -1], [-1, 0]])
    text = format_hop_connection_probability(hop_connection_probability(result, hops))

    assert "bridge hops" in text and "'module'" in text
    assert "n/a" in text
    assert text.strip().splitlines()[-1].split()[0] == "n/a"   # 到達不能は末尾
