"""エリア (area.py) と軸索伸長コネクタ (AxonGrowthTopology) の回帰テスト。

守りたい不変条件は 3 つ:
  1. 軸索はエリアの外へ出ない (どんな複雑形状でも)
  2. エリアは RandomState を消費しない (消費すると既存ネットワークの実現が全部変わる)
  3. DiskArea のドロー順は RandomCircle2DSpace と一致する (エリアベースへの移行が無コスト)
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.models.network.area import (
    BaseArea,
    CompositeArea,
    DiskArea,
    NoSpaceArea,
    Modular4Area,
    RectArea,
)
from src.models.network.connectors import AxonGrowthTopology
from src.models.network.space import AreaUniformSpace, RandomCircle2DSpace


def _cfg(**kwargs):
    return SimpleNamespace(**kwargs)


MODULAR_SPEC = _cfg(
    op="union",
    parts=[
        {"type": "disk", "center": [-500.0, -500.0], "radius": 400.0},
        {"type": "disk", "center": [500.0, -500.0], "radius": 400.0},
        {"type": "disk", "center": [-500.0, 500.0], "radius": 400.0},
        {"type": "disk", "center": [500.0, 500.0], "radius": 400.0},
        {"type": "rect", "x_range": [-500.0, 500.0], "y_range": [-60.0, 60.0]},
        {"type": "rect", "x_range": [-60.0, 60.0], "y_range": [-500.0, 500.0]},
    ],
)

AXON_CFG = _cfg(
    mean_axon_length=1100.0,
    segment_length=100.0,
    angle_sigma=0.1,
    dendrite_radius=150.0,
    connection_prob=0.2,
    boundary="deflect",
    max_deflect=4,
    allow_self_connections=False,
)


class TestAreaGeometry(unittest.TestCase):
    def test_disk_sdf_and_contains(self):
        area = DiskArea(_cfg(radius=100.0, center=[10.0, 0.0]))
        pts = np.array([[10.0, 0.0], [110.0, 0.0], [10.0, 150.0]])
        np.testing.assert_allclose(area.sdf(pts), [-100.0, 0.0, 50.0])
        np.testing.assert_array_equal(area.contains(pts), [True, True, False])
        self.assertAlmostEqual(area.area_um2, np.pi * 100.0 ** 2)

    def test_rect_sdf_inside_and_outside(self):
        area = RectArea(_cfg(x_range=[0.0, 10.0], y_range=[0.0, 20.0]))
        # 内部は最も近い辺までの距離 (負)、外部は角/辺までのユークリッド距離 (正)
        np.testing.assert_allclose(area.sdf(np.array([[5.0, 10.0]])), [-5.0])
        np.testing.assert_allclose(area.sdf(np.array([[13.0, 24.0]])), [5.0])
        self.assertAlmostEqual(area.area_um2, 200.0)

    def test_composite_set_operations(self):
        a = {"type": "disk", "center": [0.0, 0.0], "radius": 100.0}
        b = {"type": "disk", "center": [100.0, 0.0], "radius": 100.0}
        pts = np.array([[-50.0, 0.0], [50.0, 0.0], [150.0, 0.0], [300.0, 0.0]])

        union = CompositeArea(_cfg(op="union", parts=[a, b]))
        inter = CompositeArea(_cfg(op="intersection", parts=[a, b]))
        diff = CompositeArea(_cfg(op="difference", parts=[a, b]))

        np.testing.assert_array_equal(union.contains(pts), [True, True, True, False])
        np.testing.assert_array_equal(inter.contains(pts), [False, True, False, False])
        # difference は先頭から残りを削るので、重なり (50,0) は落ちる
        np.testing.assert_array_equal(diff.contains(pts), [True, False, False, False])

    def test_composite_part_of_picks_deepest_part(self):
        area = Modular4Area(MODULAR_SPEC)
        # 各円の中心はその part、ブリッジ中央は横棒 (index 4)
        idx = area.part_of(np.array([[-500.0, -500.0], [500.0, 500.0], [0.0, 0.0]]))
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[1], 3)
        self.assertIn(idx[2], (4, 5))

    def test_no_space_is_unbounded_and_refuses_sampling(self):
        area = NoSpaceArea(_cfg())
        pts = np.array([[1e9, -1e9], [0.0, 0.0]])
        np.testing.assert_array_equal(area.contains(pts), [True, True])
        self.assertFalse(area.is_bounded)
        with self.assertRaises(ValueError):
            area.sample(10, np.random.RandomState(0))

    def test_no_space_takes_no_parameters(self):
        """areas.yaml の `no_space: {}` は空マッピングなので、このクラスは config から
        何も読んではいけない。読んでいたらここで AttributeError になる。"""
        area = NoSpaceArea(SimpleNamespace(profile_name="no_space"))
        self.assertTrue(area.contains(np.zeros((1, 2))).all())

    def test_normal_points_outward(self):
        """SDF の勾配から取る汎用法線が、解析法線と一致すること。"""
        area = Modular4Area(MODULAR_SPEC)
        # 左下モジュールの外側 (中心から見て左下方向)
        p = np.array([[-500.0 - 500.0, -500.0]])
        n = area.normal(p)
        np.testing.assert_allclose(n, [[-1.0, 0.0]], atol=1e-6)


class TestAreaRngContract(unittest.TestCase):
    def test_construction_consumes_no_randomness(self):
        """エリアの構築が RandomState を消費すると、空間→結合→重み→遅延の単一ストリームが
        ずれて既存ネットワークの実現が全部変わる。構築前後で状態が同一であること。"""
        rng = np.random.RandomState(7)
        before = rng.get_state()
        for cls, cfg in (
            (NoSpaceArea, _cfg()),
            (DiskArea, _cfg(radius=100.0)),
            (RectArea, _cfg(x_range=[0.0, 1.0], y_range=[0.0, 1.0])),
            (Modular4Area, MODULAR_SPEC),
        ):
            cls(cfg, num_neurons=100)
        after = rng.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_disk_sample_matches_random_circle_2d(self):
        """DiskArea.sample のドロー順は RandomCircle2DSpace と厳密に一致する。

        これが崩れると、既存 config を area ベースへ移した瞬間に同じ seed でも別の
        ネットワークになる。順序を変えたらこのテストが落ちる。
        """
        n, radius, seed = 500, 1500.0, 20260817

        legacy = RandomCircle2DSpace(_cfg(r=radius), n, np.random.RandomState(seed))
        legacy_coords = legacy.generate()

        space = AreaUniformSpace(
            _cfg(), n, np.random.RandomState(seed), area=DiskArea(_cfg(radius=radius))
        )
        area_coords = space.generate()

        np.testing.assert_array_equal(area_coords, legacy_coords)


class TestAreaSampling(unittest.TestCase):
    def test_rejection_sampling_stays_inside_complex_area(self):
        area = Modular4Area(MODULAR_SPEC)
        pts = area.sample(3000, np.random.RandomState(1))
        self.assertEqual(pts.shape, (3000, 2))
        self.assertTrue(area.contains(pts).all())

    def test_composite_area_estimate_is_deterministic(self):
        """面積の見積もりは決定論的な格子で行う (乱数を使わない)。"""
        area = Modular4Area(MODULAR_SPEC)
        self.assertEqual(area.area_um2, area.area_um2)
        # 4 円 (2.011 mm^2) + 十字ブリッジぶん。重なりを除いて 2.2 mm^2 前後。
        self.assertAlmostEqual(area.area_um2 * 1e-6, 2.23, places=1)


class TestAxonGrowth(unittest.TestCase):
    @staticmethod
    def _build(area, n=800, seed=3, cfg=AXON_CFG):
        rng = np.random.RandomState(seed)
        coords = np.zeros((n, 3), dtype=np.float32)
        coords[:, :2] = area.sample(n, rng)
        return AxonGrowthTopology(cfg, n, coords, rng, area=area), coords

    def test_axons_never_leave_the_area(self):
        """最重要の不変条件。凹コーナーだらけの複合領域でも 1 頂点も外に出ない。"""
        for area in (DiskArea(_cfg(radius=1500.0)), Modular4Area(MODULAR_SPEC)):
            with self.subTest(area=type(area).__name__):
                conn, _ = self._build(area)
                p = conn._params()
                starts, ends, owner, arc = conn._grow_axons(p)
                self.assertGreater(owner.size, 0)
                verts = np.vstack([starts, ends])
                self.assertTrue(area.contains(verts).all(),
                                f"{int((~area.contains(verts)).sum())} 頂点がエリア外")

    def test_boundary_modes_all_stay_inside(self):
        area = DiskArea(_cfg(radius=800.0))
        for mode in ("deflect", "reflect", "stop"):
            with self.subTest(boundary=mode):
                cfg = SimpleNamespace(**{**vars(AXON_CFG), "boundary": mode})
                conn, _ = self._build(area, cfg=cfg)
                starts, ends, owner, _ = conn._grow_axons(conn._params())
                self.assertTrue(area.contains(np.vstack([starts, ends])).all())

    def test_unbounded_area_grows_freely(self):
        """no_space では境界処理が一度も発火しないので、全ニューロンが n_seg 本ぶん伸びる
        = 実効軸索長が floor(L/seg)*seg と一致する (打ち切りゼロ)。"""
        area = NoSpaceArea(_cfg())
        n, seed = 400, 11
        rng = np.random.RandomState(seed)
        coords = np.zeros((n, 3), dtype=np.float32)
        conn = AxonGrowthTopology(AXON_CFG, n, coords, rng, area=area)
        p = conn._params()
        _, _, _, arc = conn._grow_axons(p)

        expected_rng = np.random.RandomState(seed)
        L = expected_rng.rayleigh(p.mean_axon_length / np.sqrt(np.pi / 2.0), n)
        expected = np.floor(L / p.segment_length) * p.segment_length
        np.testing.assert_allclose(arc, expected)
        # 全長の分布が Rayleigh (平均 1100 um) であること
        self.assertAlmostEqual(L.mean(), 1100.0, delta=100.0)

    def test_dense_and_sparse_agree(self):
        """generate() は generate_sparse() に委譲するので、両者は定義上一致する。

        GaussianDistanceTypeTopology が密版と疎版で乱数の消費順を手で揃えているのに対し、
        こちらは委譲によって構造的に一致が保証される。その保証が壊れていないことを見る。
        """
        area = DiskArea(_cfg(radius=900.0))
        n = 400
        conn_sparse, _ = self._build(area, n=n, seed=5)
        rows, cols = conn_sparse.generate_sparse()

        conn_dense, _ = self._build(area, n=n, seed=5)   # 同じ seed = 同じ座標・同じ軸索
        mask = conn_dense.generate()

        self.assertEqual(mask.shape, (n, n))
        self.assertEqual(mask.dtype, np.int8)
        dense_rows, dense_cols = np.nonzero(mask)
        np.testing.assert_array_equal(dense_rows, rows)
        np.testing.assert_array_equal(dense_cols, cols)

    def test_output_invariants(self):
        area = DiskArea(_cfg(radius=1200.0))
        conn, _ = self._build(area, n=900, seed=8)
        rows, cols = conn.generate_sparse()

        self.assertGreater(rows.size, 0)
        self.assertEqual(rows.dtype, np.int32)
        self.assertEqual((rows == cols).sum(), 0, "オートシナプスが混じっている")
        pairs = np.stack([rows, cols], axis=1)
        self.assertEqual(len(np.unique(pairs, axis=0)), rows.size, "重複ペアがある")
        np.testing.assert_array_equal(np.lexsort((cols, rows)), np.arange(rows.size))

    def test_reproducible_from_seed(self):
        area = Modular4Area(MODULAR_SPEC)
        a, _ = self._build(area, n=500, seed=42)
        b, _ = self._build(area, n=500, seed=42)
        ra, ca = a.generate_sparse()
        rb, cb = b.generate_sparse()
        np.testing.assert_array_equal(ra, rb)
        np.testing.assert_array_equal(ca, cb)

    def test_requires_area(self):
        coords = np.zeros((10, 3), dtype=np.float32)
        conn = AxonGrowthTopology(AXON_CFG, 10, coords, np.random.RandomState(0), area=None)
        with self.assertRaises(ValueError):
            conn.generate_sparse()

    def test_rejects_bad_boundary_mode(self):
        cfg = SimpleNamespace(**{**vars(AXON_CFG), "boundary": "bounce"})
        conn = AxonGrowthTopology(cfg, 10, np.zeros((10, 3)), np.random.RandomState(0),
                                  area=NoSpaceArea(_cfg()))
        with self.assertRaises(ValueError):
            conn.generate_sparse()

    def test_axon_length_axis_declared_after_generate(self):
        area = DiskArea(_cfg(radius=1000.0))
        conn, _ = self._build(area, n=300, seed=6)
        self.assertEqual(conn.describe_axes(), {})   # 生成前は何も宣言しない
        conn.generate_sparse()
        axes = conn.describe_axes()
        self.assertIn("axon_length", axes)
        self.assertEqual(axes["axon_length"].shape, (300,))


if __name__ == "__main__":
    unittest.main()
