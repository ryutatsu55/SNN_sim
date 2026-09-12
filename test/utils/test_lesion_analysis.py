"""損傷実験のために足した解析モジュール (axons / graph / isi) と復元経路。

実データではなく**合成データで検算**する (test_beggs_plenz.py と同じ方針)。
area は `src/models` を import せずダックタイピングの fake で作る —— これは
`analysis/` 層が「area を型ではなくインタフェースで受ける」という契約そのものの検証でもある。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.analysis.axons import (  # noqa: E402
    subset_geometry, synapse_crossed_parts, synapse_path_segments)
from src.utils.analysis.connectivity import bridge_part_indices  # noqa: E402
from src.utils.analysis.graph import (  # noqa: E402
    classify_roles, clustering_coefficients, component_sizes, degree_table,
    module_participation)
from src.utils.analysis.isi import (  # noqa: E402
    cv_isi, isi_metrics, isi_per_neuron, local_variation)
from src.utils.experiments.lesion.restore import (  # noqa: E402
    WeightRestoreError, align_saved_to_coo, align_subset_to_coo)


# ======================================================================================
# ダックタイピングの fake area
# ======================================================================================

class _Rect:
    def __init__(self, x0, y0, x1, y1):
        self.box = (x0, y0, x1, y1)

    def contains(self, points):
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        x0, y0, x1, y1 = self.box
        return (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)


class _Area:
    def __init__(self, parts, names, allows):
        self.parts, self.part_names, self.part_allows_soma = parts, names, allows


class _Geometry:
    """AxonGeometry のダックタイプ。1 ニューロン 1 軸索の最小構成。"""

    def __init__(self, seg_start, seg_end, offsets, pre, post, contact_seg, contact_t):
        self.seg_start = np.asarray(seg_start, dtype=np.float64)
        self.seg_end = np.asarray(seg_end, dtype=np.float64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.pre = np.asarray(pre, dtype=np.int64)
        self.post = np.asarray(post, dtype=np.int64)
        self.contact_seg = np.asarray(contact_seg, dtype=np.int64)
        self.contact_t = np.asarray(contact_t, dtype=np.float64)


def _area_with_bridge():
    # モジュール 2 枚 (x: 0-10, 20-30) と、その間のブリッジ (x: 10-20, y: 4-6)
    return _Area(
        parts=[_Rect(0, 0, 10, 10), _Rect(20, 0, 30, 10), _Rect(10, 4, 20, 6)],
        names=["M0", "M1", "B0-1"],
        allows=[True, True, False],
    )


def test_bridge_is_the_part_where_somas_are_not_allowed():
    assert bridge_part_indices(_area_with_bridge()).tolist() == [2]


def test_bridge_needs_a_composite_area():
    with pytest.raises(ValueError):
        bridge_part_indices(_Area(parts=[], names=[], allows=[]))
    with pytest.raises(ValueError):
        # allow_soma: false の part が無い = ブリッジと呼べるものが無い
        bridge_part_indices(_Area([_Rect(0, 0, 1, 1)], ["M0"], [True]))


def test_axon_crossing_a_bridge_is_detected():
    # y=5 を x=5 -> 25 まで直進し、x=25 で接触。ブリッジ帯 (x 10-20) を必ず通る。
    geometry = _Geometry(
        seg_start=[[5, 5], [15, 5]], seg_end=[[15, 5], [25, 5]],
        offsets=[0, 2], pre=[0], post=[1], contact_seg=[1], contact_t=[1.0],
    )
    area = _area_with_bridge()
    assert synapse_crossed_parts(geometry, area, [2])[0, 0]


def test_bridge_entered_after_the_contact_is_not_counted():
    """接触より後に通ったブリッジを拾わないこと。

    判定範囲を `offsets[pre] .. contact_seg` に限る理由そのもの。軸索全体で判定すると、
    その結合とは無関係な経路で切られてしまう。
    """
    geometry = _Geometry(
        # 1 本目 (x 0->9, y=5) の途中で接触し、2 本目でブリッジへ入る
        seg_start=[[0, 5], [9, 5]], seg_end=[[9, 5], [19, 5]],
        offsets=[0, 2], pre=[0], post=[1], contact_seg=[0], contact_t=[1.0],
    )
    assert not synapse_crossed_parts(geometry, _area_with_bridge(), [2])[0, 0]


def test_contact_t_truncates_the_last_segment():
    """最終セグメントは contact_t で切る。t より先のブリッジ侵入は数えない。"""
    # セグメントは x=5 -> 25。t=0.1 なら x=7 までしか進んでおらず、ブリッジ (x>=10) 未到達。
    geometry = _Geometry(
        seg_start=[[5, 5]], seg_end=[[25, 5]],
        offsets=[0, 1], pre=[0], post=[1], contact_seg=[0], contact_t=[0.1],
    )
    starts, ends = synapse_path_segments(geometry, 0)
    assert ends[-1][0] == pytest.approx(7.0)
    assert not synapse_crossed_parts(geometry, _area_with_bridge(), [2])[0, 0]


def test_more_samples_never_lose_a_detection():
    """サンプル数を増やして偽陽性は出ない (単調に検出が増えるだけ)。"""
    geometry = _Geometry(
        seg_start=[[5, 5]], seg_end=[[25, 5]],
        offsets=[0, 1], pre=[0], post=[1], contact_seg=[0], contact_t=[1.0],
    )
    area = _area_with_bridge()
    coarse = synapse_crossed_parts(geometry, area, [2], samples=8)[0, 0]
    fine = synapse_crossed_parts(geometry, area, [2], samples=256)[0, 0]
    assert fine >= coarse


# ======================================================================================
# graph.py
# ======================================================================================

def _two_module_graph():
    """M0={0,1,2} / M1={3,4,5}、モジュール内は全結合、間は 2<->3 の 1 本だけ。"""
    edges = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
             (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4), (2, 3), (3, 2)]
    row = np.array([e[0] for e in edges])
    col = np.array([e[1] for e in edges])
    labels = np.array(["M0", "M0", "M0", "M1", "M1", "M1"])
    return row, col, labels


def test_degree_counts():
    row, col, _ = _two_module_graph()
    table = degree_table(row, col, 6)
    assert table["out_degree"].tolist() == [2, 2, 3, 3, 2, 2]
    assert table["in_degree"].tolist() == [2, 2, 3, 3, 2, 2]


def test_participation_is_zero_inside_one_module_and_positive_across():
    row, col, labels = _two_module_graph()
    p = module_participation(row, col, 6, labels)["participation"]
    assert p[0] == pytest.approx(0.0)
    assert p[1] == pytest.approx(0.0)
    assert p[2] > 0.0        # モジュールをまたぐ辺を持つ
    assert p[3] > 0.0


def test_isolated_neurons_get_zero_not_nan():
    """次数 0 を NaN にしない。「孤立している」と「計算できない」を混同させないため。"""
    result = module_participation(np.array([0]), np.array([1]), 3, np.array(["A", "A", "B"]))
    assert np.all(np.isfinite(result["participation"]))
    assert np.all(np.isfinite(result["within_module_z"]))
    assert result["participation"][2] == 0.0


def test_within_module_z_is_zero_when_all_degrees_match():
    """モジュール内の次数が全部同じ (sd=0) でも inf/NaN にならないこと。"""
    row, col, labels = _two_module_graph()
    z = module_participation(row, col, 6, labels)["within_module_z"]
    assert np.all(np.isfinite(z))


def test_clustering_is_one_for_a_triangle():
    row = np.array([0, 1, 0, 2, 1, 2])
    col = np.array([1, 0, 2, 0, 2, 1])
    assert clustering_coefficients(row, col, 3).tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_cutting_the_bridge_splits_the_network():
    """損傷でネットワークが割れたことを最も直接に示す量。"""
    row, col, _ = _two_module_graph()
    assert component_sizes(row, col, 6)["num_weak_components"] == 1.0
    keep = ~(((row == 2) & (col == 3)) | ((row == 3) & (col == 2)))
    after = component_sizes(row[keep], col[keep], 6)
    assert after["num_weak_components"] == 2.0
    assert after["largest_weak_fraction"] == pytest.approx(0.5)


def test_role_classification_boundaries():
    p = np.array([0.0, 0.7, 0.9, 0.7])
    z = np.array([3.0, 3.0, 3.0, 0.0])
    roles = classify_roles(p, z).tolist()
    assert roles == ["provincial_hub", "connector_hub", "kinless_hub", "non_hub_connector"]


# ======================================================================================
# isi.py
# ======================================================================================

def test_periodic_train_has_zero_variability():
    times = np.arange(0.0, 1000.0, 10.0)
    ids = np.zeros(times.size, dtype=np.int64)
    values, offsets = isi_per_neuron(times, ids, 1)
    assert values.size == times.size - 1
    assert cv_isi(values, offsets)[0] == pytest.approx(0.0)
    assert local_variation(values, offsets)[0] == pytest.approx(0.0)


def test_poisson_train_has_cv_near_one():
    rng = np.random.default_rng(0)
    times = np.cumsum(rng.exponential(10.0, 4000))
    ids = np.zeros(times.size, dtype=np.int64)
    values, offsets = isi_per_neuron(times, ids, 1)
    assert cv_isi(values, offsets)[0] == pytest.approx(1.0, abs=0.1)
    assert local_variation(values, offsets)[0] == pytest.approx(1.0, abs=0.15)


def test_isi_never_spans_two_neurons():
    """ニューロンをまたぐ差分を ISI に混ぜないこと。"""
    times = np.array([0.0, 10.0, 1000.0, 1010.0])
    ids = np.array([0, 0, 1, 1])
    values, offsets = isi_per_neuron(times, ids, 2)
    assert values.tolist() == [10.0, 10.0]
    assert offsets.tolist() == [0, 1, 2]


def test_sparse_neurons_are_excluded_from_cv_but_counted_as_silent():
    """スパイク 2 発以下を CV に 0 として混ぜない (黙ったのを「規則的」と読ませない)。"""
    times = np.array([5.0])
    ids = np.array([0])
    metrics = isi_metrics(times, ids, 4, 1000.0)
    assert metrics["silent_fraction"] == pytest.approx(0.75)
    assert metrics["num_neurons_with_isi"] == 0
    assert np.isnan(metrics["cv_isi_mean"])


# ======================================================================================
# restore.py — 並び順の取り違えを防ぐ
# ======================================================================================

class _Coo:
    def __init__(self, row, col, shape):
        self.row, self.col, self.shape = np.asarray(row), np.asarray(col), shape


def test_restore_survives_a_permuted_saved_order():
    """保存側の並びが違っても (pre,post) join で正しく戻ること。これがこの層の存在理由。"""
    row = np.array([0, 0, 1, 2, 3])
    col = np.array([1, 3, 2, 0, 1])
    coo = _Coo(row, col, (4, 4))
    perm = np.array([3, 0, 4, 2, 1])
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    restored = align_saved_to_coo(row[perm], col[perm], values[perm], coo)
    assert restored.tolist() == pytest.approx(values.tolist())


def test_restore_rejects_a_different_network():
    coo = _Coo([0, 1], [1, 2], (4, 4))
    with pytest.raises(WeightRestoreError, match="本数"):
        align_saved_to_coo([0], [1], [0.1], coo)
    with pytest.raises(WeightRestoreError, match="長さ"):
        align_saved_to_coo([0, 1], [1, 2], [0.1], coo)
    with pytest.raises(WeightRestoreError, match="保存側にありません"):
        align_saved_to_coo([0, 3], [1, 3], [0.1, 0.2], coo)
    with pytest.raises(WeightRestoreError, match="複数"):
        align_saved_to_coo([0, 0], [1, 1], [0.1, 0.2], coo)


def test_restore_key_does_not_overflow_int32():
    """row*N+col は int64 で計算すること。int32 だと大きな N で折り返って別ペアと衝突する。"""
    n = 100000
    row = np.array([0, 50000, 99999], dtype=np.int32)
    col = np.array([1, 50000, 99999], dtype=np.int32)
    assert row.astype(np.int64).max() * n > 2 ** 31   # 折り返しうる領域であることの確認
    coo = _Coo(row, col, (n, n))
    values = np.array([0.1, 0.2, 0.3])
    restored = align_saved_to_coo(row, col, values, coo)
    assert restored.tolist() == pytest.approx(values.tolist())


# ======================================================================================
# subset_geometry — 切断後の図が切断前を映さないこと
# ======================================================================================

def _two_synapse_geometry():
    """2 ニューロン、それぞれ 1 本ずつ軸索とシナプスを持つ最小構成。"""
    return _Geometry(
        seg_start=[[0, 5], [10, 5], [0, 1], [10, 1]],
        seg_end=[[10, 5], [20, 5], [10, 1], [20, 1]],
        offsets=[0, 2, 4], pre=[0, 1], post=[1, 0],
        contact_seg=[1, 3], contact_t=[0.5, 0.5],
    )


def test_subset_geometry_drops_only_the_synapse_arrays():
    geometry = _two_synapse_geometry()
    subset = subset_geometry(geometry, np.array([True, False]))
    assert subset.pre.tolist() == [0]
    assert subset.post.tolist() == [1]
    assert subset.contact_seg.tolist() == [1]
    # 軸索そのものは切っていないので、下敷きの形は変わらない
    assert subset.seg_start.shape == geometry.seg_start.shape
    assert subset.offsets.tolist() == geometry.offsets.tolist()


def test_subset_geometry_keeps_contact_seg_indices_valid():
    """セグメント配列を保つので contact_seg の添字は有効なまま。"""
    geometry = _two_synapse_geometry()
    subset = subset_geometry(geometry, np.array([False, True]))
    assert subset.contact_seg.max() < subset.seg_start.shape[0]
    starts, _ends = synapse_path_segments(subset, 0)
    assert starts.shape[0] > 0


def test_subset_geometry_does_not_mutate_the_original():
    geometry = _two_synapse_geometry()
    subset_geometry(geometry, np.array([True, False]))
    assert geometry.pre.size == 2


def test_cut_synapses_are_gone_from_the_subset():
    """図が切断前を映していた不具合の回帰テスト。

    ブリッジを通った結合を切ったなら、切断後の幾何で同じ判定をかけて 0 本でなければ
    `axon_network()` はその結合を描いてしまう。
    """
    geometry = _Geometry(
        seg_start=[[5, 5], [0, 9]], seg_end=[[25, 5], [8, 9]],
        offsets=[0, 1, 2], pre=[0, 1], post=[1, 0], contact_seg=[0, 1], contact_t=[1.0, 1.0],
    )
    area = _area_with_bridge()
    crossed = synapse_crossed_parts(geometry, area, [2]).ravel()
    assert crossed.tolist() == [True, False]      # 1 本目だけがブリッジを通る
    subset = subset_geometry(geometry, ~crossed)
    assert not synapse_crossed_parts(subset, area, [2]).any()


# ======================================================================================
# align_subset_to_coo — 切断前後で本数が違う記録を同じ図に載せるための引き当て
# ======================================================================================

def test_subset_alignment_maps_pre_lesion_values_onto_survivors():
    """baseline probe (切断前・全シナプス) を、生き残ったシナプスの並びへ引き当てる。"""
    saved_row = np.array([0, 0, 1, 2, 3])
    saved_col = np.array([1, 3, 2, 0, 1])
    saved_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    keep = np.array([True, False, True, False, True])   # 2 本切断
    aligned = align_subset_to_coo(saved_row, saved_col, saved_values,
                                  saved_row[keep], saved_col[keep], 4)
    assert aligned.tolist() == pytest.approx([0.1, 0.3, 0.5])


def test_subset_alignment_survives_a_permuted_saved_order():
    saved_row = np.array([0, 0, 1, 2, 3])
    saved_col = np.array([1, 3, 2, 0, 1])
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    perm = np.array([4, 1, 0, 3, 2])
    keep = np.array([True, False, True, False, True])
    aligned = align_subset_to_coo(saved_row[perm], saved_col[perm], values[perm],
                                  saved_row[keep], saved_col[keep], 4)
    assert aligned.tolist() == pytest.approx([0.1, 0.3, 0.5])


def test_subset_alignment_rejects_a_target_that_is_not_a_subset():
    with pytest.raises(WeightRestoreError, match="部分集合"):
        align_subset_to_coo([0, 1], [1, 2], [0.1, 0.2], [3], [3], 4)
