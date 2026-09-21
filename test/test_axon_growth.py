"""エリア (area.py) と軸索伸長コネクタ (AxonGrowthTopology) の回帰テスト。

守りたい不変条件は 4 つ:
  1. 軸索はエリアの外へ出ない (どんな複雑形状でも)。**端点だけでなく線分全体が**内部で
     あること — 端点しか見ないとセグメント長より狭い空隙を飛び越えてしまう
  2. 互いに接触していない部分領域の間に結合は生まれない (軸索も樹状突起も空隙を越えない)
  3. エリアは RandomState を消費しない (消費すると既存ネットワークの実現が全部変わる)
  4. DiskArea のドロー順は RandomCircle2DSpace と一致する (エリアベースへの移行が無コスト)
"""

import sys
import tempfile
import unittest
from dataclasses import fields
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
from src.models.network.connectors import AxonGeometry, AxonGrowthTopology
from src.models.network.space import AreaUniformSpace, RandomCircle2DSpace
from src.core.output_manager import AXONS_NAME


def _cfg(**kwargs):
    return SimpleNamespace(**kwargs)



def _area_profile(name):
    """areas.yaml のプロファイルをそのまま読む (テスト側で形を書き写さないため)。"""
    import yaml
    with open(root_path / "configs" / "components" / "areas.yaml", encoding="utf-8") as fh:
        return _cfg(**yaml.safe_load(fh)[name])


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

    def test_segment_inside_detects_a_gap_between_parts(self):
        """非連結な複合領域では、両端点が内部でも途中が外に出る線分を弾くこと。

        modular_4 の円 (中心 (-500,-500), r=400) と横棒 (y in [-60,60]) の間には
        40 um の空隙がある。両端がそれぞれの内部にある線分は `contains` の端点判定を
        通ってしまうが、`segment_inside` は落とさなければならない。
        """
        area = Modular4Area(MODULAR_SPEC)
        a = np.array([[-500.0, -150.0],    # 円の内部 (境界まで 50 um)
                      [-500.0, -500.0]])   # 円の中心
        b = np.array([[-500.0, -30.0],     # 横棒の内部
                      [-400.0, -500.0]])   # 同じ円の内部
        np.testing.assert_array_equal(area.contains(a), [True, True])
        np.testing.assert_array_equal(area.contains(b), [True, True])
        # 1 本目は空隙をまたぐので False、2 本目は円の中で閉じているので True
        np.testing.assert_array_equal(area.segment_inside(a, b), [False, True])

    def test_convex_segment_inside_matches_sampling(self):
        """凸形状の解析的オーバーライドが、基底のサンプリング実装と一致すること。

        `DiskArea` / `RectArea` は「端点が内部なら線分も内部」に短絡している。凸なので
        厳密なはずだが、取り違えるとサンプリング版とズレる。
        """
        rng = np.random.RandomState(0)
        for area in (DiskArea(_cfg(radius=300.0, center=[10.0, -20.0])),
                     RectArea(_cfg(x_range=[0.0, 400.0], y_range=[-100.0, 250.0]))):
            with self.subTest(area=type(area).__name__):
                lo, hi = area.bounds
                a = rng.uniform(lo - 50.0, hi + 50.0, size=(500, 2))
                b = rng.uniform(lo - 50.0, hi + 50.0, size=(500, 2))
                np.testing.assert_array_equal(
                    area.segment_inside(a, b),
                    BaseArea.segment_inside(area, a, b),
                )

    def test_no_space_segment_inside_is_always_true(self):
        area = NoSpaceArea(_cfg())
        a = np.array([[0.0, 0.0], [1e9, 1e9]])
        b = np.array([[1e9, -1e9], [-1e9, 0.0]])
        np.testing.assert_array_equal(area.segment_inside(a, b), [True, True])

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

    # ----------------------------------------- allow_soma (soma 配置領域と軸索領域の分離)

    def test_soma_area_defaults_to_the_area_itself(self):
        """allow_soma を書かなければ soma 配置領域は領域そのもの。

        **同一オブジェクトであること**が要点。これが `sample()` の乱数消費を従来と
        1 draw も変えない = 既存 config の実現が不変であることの担保になる。
        """
        for area in (DiskArea(_cfg(radius=1.0)),
                     RectArea(_cfg(x_range=[0.0, 1.0], y_range=[0.0, 1.0])),
                     NoSpaceArea(_cfg()),
                     CompositeArea(MODULAR_SPEC)):
            with self.subTest(area=type(area).__name__):
                self.assertIs(area.soma_area, area)

    def test_allow_soma_false_excludes_a_part(self):
        """allow_soma: false の part は soma 配置領域から外れる (領域の形自体は変わらない)。"""
        area = CompositeArea(_cfg(op="union", parts=[
            {"type": "disk", "name": "M0", "center": [0.0, 0.0], "radius": 10.0},
            {"type": "rect", "name": "B0", "allow_soma": False,
             "x_range": [8.0, 40.0], "y_range": [-2.0, 2.0]},
        ]))
        self.assertEqual(area.part_allows_soma, [True, False])
        # 形そのもの (軸索が動ける範囲) は不変 — 通路の先端は今も領域内
        self.assertTrue(area.contains(np.array([[35.0, 0.0]]))[0])

        region = area.soma_area
        self.assertIsNot(region, area)
        self.assertEqual(region.part_names, ["M0"])   # 名前は親から引き継ぐ
        self.assertIs(area.soma_area, region)         # キャッシュされる

        pts = region.sample(500, np.random.RandomState(0))
        self.assertTrue(area.contains(pts).all(), "soma が元の領域の外に出た")
        self.assertFalse((pts[:, 0] > 10.0).any(), "soma が通路に落ちた")

    def test_allow_soma_requires_union_and_at_least_one_part(self):
        """union 以外では「その part を抜いた領域」が定義できない。全除外も許さない。"""
        excluded = {"type": "rect", "allow_soma": False,
                    "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
        disk = {"type": "disk", "center": [0.0, 0.0], "radius": 1.0}
        for op in ("intersection", "difference"):
            with self.subTest(op=op):
                with self.assertRaises(ValueError):
                    CompositeArea(_cfg(op=op, parts=[disk, excluded])).soma_area
        with self.assertRaises(ValueError):
            CompositeArea(_cfg(op="union", parts=[excluded])).soma_area

    def test_soma_area_does_not_change_axon_confinement(self):
        """soma 配置領域を分けても、軸索側が使うメソッドの答えは 1 つも変わらない。"""
        parts = [{"type": "disk", "name": "M0", "center": [0.0, 0.0], "radius": 10.0},
                 {"type": "rect", "name": "B0", "x_range": [8.0, 40.0], "y_range": [-2.0, 2.0]}]
        plain = CompositeArea(_cfg(op="union", parts=parts))
        gated = CompositeArea(_cfg(op="union", parts=[
            parts[0], {**parts[1], "allow_soma": False}]))
        pts = np.random.RandomState(0).uniform(-45.0, 45.0, size=(400, 2))
        np.testing.assert_allclose(gated.sdf(pts), plain.sdf(pts))
        np.testing.assert_array_equal(gated.segment_inside(pts[:200], pts[200:]),
                                      plain.segment_inside(pts[:200], pts[200:]))

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

    def test_axon_segments_never_cross_a_void(self):
        """頂点だけでなく**線分の内部**も領域内であること。

        modular_4 は円とブリッジが 40 um 離れた 5 つの孤立成分なので、セグメント長
        100 um の軸索は端点判定だけだと空隙を飛び越えられる (修正前は 891 本中 104 本が
        実際に飛んでいた)。中点が外に出ていないことも併せて見る。
        """
        area = Modular4Area(MODULAR_SPEC)
        conn, _ = self._build(area)
        starts, ends, owner, _ = conn._grow_axons(conn._params())
        self.assertGreater(owner.size, 0)

        mid_out = ~area.contains(0.5 * (starts + ends))
        self.assertFalse(mid_out.any(), f"{int(mid_out.sum())} セグメントの中点がエリア外")
        inside = area.segment_inside(starts, ends)
        self.assertTrue(inside.all(), f"{int((~inside).sum())} セグメントが領域外を通過")

    def test_no_synapses_between_isolated_components(self):
        """接触していない部分領域の間には結合が生まれないこと。

        modular_4 の 6 parts は「4 円 + 十字 (矩形 2 枚は互いに交差)」= 5 連結成分。
        軸索の空隙ジャンプと、樹状突起半径 (150 um) が空隙 (40 um) を越えて届くことの
        両方を塞げていれば 0 件になる (修正前は 38404 本中 4240 本 = 11%)。
        """
        area = Modular4Area(MODULAR_SPEC)
        conn, coords = self._build(area)
        rows, cols = conn.generate_sparse()
        self.assertGreater(rows.size, 0)

        # part index -> 連結成分 index。part 4, 5 (十字の 2 枚) は交差しているので 1 成分。
        component = np.minimum(area.part_of(coords[:, :2]), 4)
        crossing = component[rows] != component[cols]
        self.assertFalse(
            crossing.any(),
            f"孤立成分をまたぐ結合が {int(crossing.sum())} / {rows.size} 本ある",
        )

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


class TestFirstExit(unittest.TestCase):
    """`BaseArea.first_exit()` — 壁までどこまで進めるか。advance-and-slide の土台。"""

    def test_returns_one_when_the_segment_stays_inside(self):
        area = DiskArea(_cfg(radius=100.0))
        np.testing.assert_allclose(area.first_exit([[0.0, 0.0]], [[50.0, 0.0]]), [1.0])

    def test_finds_the_crossing_point(self):
        area = DiskArea(_cfg(radius=100.0))
        t = area.first_exit([[0.0, 0.0]], [[200.0, 0.0]])[0]
        self.assertAlmostEqual(200.0 * t, 100.0, delta=0.1)

    def test_prefix_is_always_inside(self):
        """返した t までの部分線分は `segment_inside` が True — でなければ、伸ばしてよいと
        判断した長さが後の検証で外と判定され、軸索が領域外に出る。"""
        area = Modular4Area(MODULAR_SPEC)
        rng = np.random.RandomState(3)
        a = area.sample(400, rng)
        b = a + rng.normal(0.0, 200.0, (400, 2))
        t = area.first_exit(a, b)
        self.assertTrue(area.segment_inside(a, a + t[:, None] * (b - a)).all())
        # 出ない線分は 1.0 のまま、出る線分は必ず 1 未満
        crosses = ~area.segment_inside(a, b)
        self.assertTrue((t[~crosses] == 1.0).all())
        self.assertTrue((t[crosses] < 1.0).all())
        self.assertGreater(crosses.sum(), 0, "壁に当たる線分が 1 本も無い")

    def test_unbounded_area_never_exits(self):
        area = NoSpaceArea(_cfg())
        np.testing.assert_array_equal(area.first_exit([[0.0, 0.0]], [[1e9, 1e9]]), [1.0])


class TestAdvanceAndSlide(unittest.TestCase):
    """壁に当たったときの挙動。**交点まで進んでから**曲がり、残りを壁沿いに使う。"""

    @staticmethod
    def _grow(area, n=400, seed=3, **over):
        """伸長させ、(conn, params, (start, end, owner, arc), 名目全長) を返す。

        名目全長は「壁が無ければ伸びたはずの長さ」= floor(L/seg)*seg。座標生成で消費された
        ぶんだけ rng を進めてから同じ列を引き直すことで、実際に使われた値と一致させる。
        """
        cfg = SimpleNamespace(**{**vars(AXON_CFG), **over})
        rng = np.random.RandomState(seed)
        coords = np.zeros((n, 3), dtype=np.float64)
        coords[:, :2] = area.sample(n, rng) if area.is_bounded else 0.0
        conn = AxonGrowthTopology(cfg, n, coords, rng, area=area)
        p = conn._params()

        replay = np.random.RandomState(seed)
        if area.is_bounded:
            area.sample(n, replay)
        nominal = np.floor(
            replay.rayleigh(p.mean_axon_length / np.sqrt(np.pi / 2.0), n) / p.segment_length
        ) * p.segment_length
        return conn, p, conn._grow_axons(p), nominal

    def test_convex_walls_never_truncate_an_axon(self):
        """凸な壁は必ず滑れるので、軸索は名目全長ぶん伸びきる。

        旧実装は `base` から全長を引き直していたため、曲面の接線が領域外へ出て
        「偏向したのに外」となり、打ち切られる軸索が出ていた (円 r=400 で 28%)。
        """
        for area in (DiskArea(_cfg(radius=400.0)),
                     RectArea(_cfg(x_range=[0.0, 800.0], y_range=[0.0, 800.0]))):
            with self.subTest(area=type(area).__name__):
                _, _, (_, _, owner, arc), nominal = self._grow(area, n=300, seed=11)
                self.assertGreater(owner.size, 0)
                self.assertGreater(arc.sum() / nominal.sum(), 0.95)
                short = arc < 0.5 * nominal
                self.assertEqual(short.sum(), 0,
                                 f"{int(short.sum())} 本が半分も伸びていない")

    def test_a_segment_never_exceeds_the_step_length(self):
        """壁で複数本に割れても、1 本のセグメントが segment_length を超えることはない
        (壁に当たっても速度は落ちないが、増えもしない)。"""
        area = Modular4Area(MODULAR_SPEC)
        _, p, (ss, se, owner, arc), nominal = self._grow(area, n=300, seed=5)
        self.assertGreater(owner.size, 0)
        self.assertTrue((np.linalg.norm(se - ss, axis=1) <= p.segment_length + 1e-6).all())
        self.assertTrue((arc <= nominal + 1e-6).all(), "名目より長く伸びた軸索がある")

    def test_turns_are_gentle_in_a_smooth_area(self):
        """滑らかな壁なら、壁沿いに滑るので 90 度級の折れは生じない。

        旧実装は候補終点で法線を取り、壁に着く前に向きを変えていたため、軸に平行な壁では
        向きが軸に張り付いて折れ角の中央値が 90 度になっていた (階段状の軌跡)。
        """
        area = RectArea(_cfg(x_range=[0.0, 600.0], y_range=[0.0, 600.0]))
        _, _, (ss, se, owner, _), _ = self._grow(area, n=200, seed=7)
        d = se - ss
        theta = np.arctan2(d[:, 1], d[:, 0])
        same = owner[1:] == owner[:-1]
        turn = np.degrees(np.abs(np.arctan2(np.sin(theta[1:] - theta[:-1]),
                                            np.cos(theta[1:] - theta[:-1]))))[same]
        self.assertGreater(turn.size, 100)
        self.assertLess(np.median(turn), 20.0, f"折れ角の中央値が {np.median(turn):.1f} 度")

    def test_stop_mode_ends_at_the_wall(self):
        """boundary: stop は壁**まで**伸ばして止まる (ステップごと捨てない)。"""
        area = DiskArea(_cfg(radius=300.0))
        _, _, (ss, se, owner, arc), _ = self._grow(area, n=200, seed=9, boundary="stop")
        self.assertGreater(owner.size, 0)
        self.assertTrue(area.contains(np.vstack([ss, se])).all())
        # 壁で止まった軸索は最後の頂点が境界上にある
        last = np.nonzero(np.diff(np.concatenate([owner, [-1]])) != 0)[0]
        touching = np.abs(area.sdf(se[last])) < 1e-2
        self.assertGreater(touching.sum(), 0, "壁で止まった軸索が 1 本も無い")


class TestInvasionLottery(unittest.TestCase):
    """結合の抽選は**樹状突起円への侵入 1 回につき 1 回** (Sumi 2025 §2.8 / Orlandi 2013)。

    セグメント単位で引くと、同じ侵入を刻んだ本数だけ試行してしまい、実効確率が
    segment_length に依存する。
    """

    AREA = DiskArea(_cfg(radius=800.0))
    N = 900

    def _coords(self):
        coords = np.zeros((self.N, 3), dtype=np.float64)
        coords[:, :2] = self.AREA.sample(self.N, np.random.RandomState(21))
        return coords

    def _run(self, coords, seed=4, **over):
        cfg = SimpleNamespace(**{**vars(AXON_CFG), **over})
        conn = AxonGrowthTopology(cfg, self.N, coords, np.random.RandomState(seed),
                                  area=self.AREA)
        return conn.generate_sparse()

    def test_degree_matches_probability_times_invasions(self):
        """出次数 = connection_prob x 侵入数。p=1.0 が幾何的な侵入数そのもの。"""
        coords = self._coords()
        invasions = self._run(coords, connection_prob=1.0)[0].size
        for prob in (0.2, 0.5):
            with self.subTest(prob=prob):
                got = self._run(coords, connection_prob=prob)[0].size
                self.assertAlmostEqual(got / invasions, prob, delta=0.02)

    def test_degree_barely_depends_on_segment_length(self):
        """刻みを 1/4 にしても出次数はほぼ変わらない (セグメント単位だと約 2 倍に膨らむ)。"""
        coords = self._coords()
        coarse = self._run(coords, segment_length=100.0)[0].size
        fine = self._run(coords, segment_length=25.0)[0].size
        self.assertLess(abs(fine - coarse) / coarse, 0.15,
                        f"segment_length で出次数が変わりすぎ: {coarse} -> {fine}")

    def test_block_size_does_not_change_the_network(self):
        """_SEGMENT_BLOCK は性能のための内部都合。実現を変えてはいけない。"""
        coords = self._coords()
        original = AxonGrowthTopology._SEGMENT_BLOCK
        try:
            AxonGrowthTopology._SEGMENT_BLOCK = original
            ref = self._run(coords)
            for block in (512, 64, 7):
                with self.subTest(block=block):
                    AxonGrowthTopology._SEGMENT_BLOCK = block
                    got = self._run(coords)
                    np.testing.assert_array_equal(got[0], ref[0])
                    np.testing.assert_array_equal(got[1], ref[1])
        finally:
            AxonGrowthTopology._SEGMENT_BLOCK = original


class TestAxonGeometry(unittest.TestCase):
    """`axon_geometry()` — 折れ線と接触位置の記録。

    これは**結合の記録であって結合そのものではない**。可視化 (plotting.axon_network) と
    解析のためだけに残すので、乱数を 1 つも消費してはいけない (消費すると同じ seed の
    既存ネットワークが別物になる)。ここでは記録が結合と整合していることを見る。
    """

    @staticmethod
    def _built(n=400, seed=7, cfg=AXON_CFG):
        area = Modular4Area(MODULAR_SPEC)
        conn, coords = TestAxonGrowth._build(area, n=n, seed=seed, cfg=cfg)
        rows, cols = conn.generate_sparse()
        return area, conn, coords, rows, cols

    def test_absent_before_generate(self):
        area = DiskArea(_cfg(radius=1000.0))
        conn, _ = TestAxonGrowth._build(area, n=100, seed=1)
        self.assertIsNone(conn.axon_geometry())

    def test_polyline_starts_at_soma_and_is_continuous(self):
        _, conn, coords, _, _ = self._built()
        g = conn.axon_geometry()

        counts = np.bincount(g.seg_owner, minlength=conn.num_neurons)
        np.testing.assert_array_equal(np.diff(g.offsets), counts)
        self.assertEqual(g.offsets[-1], g.seg_owner.size)

        for i in np.nonzero(counts)[0]:
            lo, hi = g.offsets[i], g.offsets[i + 1]
            # 区間が本当にその owner だけで埋まっている (= (owner, step) 順)
            self.assertTrue((g.seg_owner[lo:hi] == i).all())
            np.testing.assert_allclose(g.seg_start[lo], coords[i, :2], atol=1e-6)
            # 連続するセグメントは端点を共有する = 折れ線として繋がっている
            np.testing.assert_allclose(g.seg_start[lo + 1:hi], g.seg_end[lo:hi - 1])

    def test_pairs_match_generate_sparse(self):
        _, conn, _, rows, cols = self._built()
        g = conn.axon_geometry()
        np.testing.assert_array_equal(g.pre, rows)
        np.testing.assert_array_equal(g.post, cols)
        self.assertEqual(g.contact_seg.size, rows.size)
        self.assertEqual(g.contact_t.size, rows.size)

    def test_contact_segment_belongs_to_the_presynaptic_axon(self):
        """接触セグメントは必ず pre 自身の軸索の中にある (= 経路が pre から始まる)。"""
        _, conn, _, _, _ = self._built()
        g = conn.axon_geometry()
        self.assertTrue((g.contact_seg >= g.offsets[g.pre]).all())
        self.assertTrue((g.contact_seg < g.offsets[g.pre + 1]).all())
        self.assertTrue(((0.0 <= g.contact_t) & (g.contact_t <= 1.0)).all())

    def test_contact_point_reaches_the_postsynaptic_soma(self):
        """接触点は post 細胞体から dendrite_radius 以内にあり、かつ見通しが立つ。

        後者は生成時の条件そのもの (CLAUDE.md #17)。記録した接触点が本物であることの検算。
        """
        area, conn, coords, _, _ = self._built()
        g = conn.axon_geometry()
        p = conn._params()
        a, b = g.seg_start[g.contact_seg], g.seg_end[g.contact_seg]
        contact = a + g.contact_t[:, None] * (b - a)
        soma = coords[g.post, :2].astype(np.float64)

        distance = np.linalg.norm(contact - soma, axis=1)
        self.assertLessEqual(distance.max(), p.dendrite_radius + 1e-6)
        self.assertTrue(area.segment_inside(contact, soma).all(),
                        "接触点から細胞体までが領域の外を通っている")

    def test_contact_is_the_earliest_touching_segment(self):
        """dedup は**最初に採用された接触**を残す (np.unique の安定ソート依存)。

        connection_prob=1.0 なら「採用された」= 「接触した」なので、記録された
        セグメントはその相手に触れた最初のセグメントでなければならない。
        """
        cfg = SimpleNamespace(**{**vars(AXON_CFG), "connection_prob": 1.0})
        area, conn, coords, _, _ = self._built(n=150, seed=4, cfg=cfg)
        g = conn.axon_geometry()
        p = conn._params()
        self.assertGreater(g.pre.size, 0)

        for k in range(0, g.pre.size, 7):        # 全数だと O(M*S) なので間引く
            pre, post, seg = int(g.pre[k]), int(g.post[k]), int(g.contact_seg[k])
            earlier = np.arange(g.offsets[pre], seg)
            if earlier.size == 0:
                continue
            a, b = g.seg_start[earlier], g.seg_end[earlier]
            ab = b - a
            ap = coords[post, :2].astype(np.float64) - a
            t = np.clip((ap * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
            perp = ap - t[:, None] * ab
            near = np.linalg.norm(perp, axis=1) <= p.dendrite_radius
            if near.any():
                contact = a[near] + t[near, None] * ab[near]
                visible = area.segment_inside(
                    contact, np.repeat(coords[post, :2].astype(np.float64)[None, :],
                                       int(near.sum()), axis=0))
                self.assertFalse(visible.any(),
                                 f"pre={pre} post={post}: seg {seg} より前に接触がある")


class TestAxonGeometryIO(unittest.TestCase):
    """`AxonGeometry.save()` / `load()` — 軸索の折れ線を run ディレクトリへ残す。

    損傷実験は「その run で実際に伸びた軸索」が対象なので、seed から再現できることに
    頼らず記録として残す (エリア・segment_length・contains() の許容差のどれかが変われば
    別の軌跡になる)。ここで守るのは **往復で 1 ビットも変わらないこと**と、
    読み戻したものが描画側のインタフェースをそのまま満たすこと。
    """

    @staticmethod
    def _geometry():
        _, conn, coords, _, _ = TestAxonGeometry._built(n=120, seed=13)
        return conn.axon_geometry(), coords

    def test_save_load_round_trip(self):
        """8 本すべてが値も dtype も一致すること。"""
        geometry, _ = self._geometry()
        with tempfile.TemporaryDirectory() as tmp:
            path = geometry.save(Path(tmp) / AXONS_NAME)
            self.assertTrue(path.exists())
            restored = AxonGeometry.load(path)
        for field in fields(AxonGeometry):
            original, loaded = getattr(geometry, field.name), getattr(restored, field.name)
            with self.subTest(array=field.name):
                np.testing.assert_array_equal(loaded, original)
                self.assertEqual(loaded.dtype, original.dtype)

    def test_save_creates_parent_directories(self):
        """run ディレクトリがまだ無くても書けること (save_axes と同じ作法)。"""
        geometry, _ = self._geometry()
        with tempfile.TemporaryDirectory() as tmp:
            path = geometry.save(Path(tmp) / "run" / "data" / AXONS_NAME)
            self.assertTrue(path.exists())

    def test_npz_needs_no_pickle(self):
        """allow_pickle=False で読めること (object 配列が混ざっていない)。"""
        geometry, _ = self._geometry()
        with tempfile.TemporaryDirectory() as tmp:
            path = geometry.save(Path(tmp) / AXONS_NAME)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(sorted(data.files),
                                 sorted(f.name for f in fields(AxonGeometry)))

    def test_load_rejects_a_file_missing_arrays(self):
        """配列が欠けた npz は、何が無いかを挙げて弾くこと (黙って部分復元しない)。"""
        geometry, _ = self._geometry()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / AXONS_NAME
            arrays = {f.name: getattr(geometry, f.name) for f in fields(AxonGeometry)}
            arrays.pop("contact_seg")
            np.savez_compressed(path, **arrays)
            with self.assertRaises(ValueError) as caught:
                AxonGeometry.load(path)
            self.assertIn("contact_seg", str(caught.exception))

    def test_loaded_geometry_still_drives_the_figure(self):
        """読み戻した幾何で axon_network() が描けること。

        `src/utils/plotting` は幾何をダックタイピングで読むので、往復してもその
        インタフェース (seg_start / offsets / contact_seg …) が保たれているかは
        実際に描かせるのが一番確実。
        """
        import matplotlib
        matplotlib.use("Agg")
        from src.utils.plotting.network import axon_network

        geometry, coords = self._geometry()
        # axon_network が触るのは config.network.space だけ (test_area_plotting と同じ)。
        config = _cfg(network=_cfg(space=_cfg()))
        with tempfile.TemporaryDirectory() as tmp:
            restored = AxonGeometry.load(geometry.save(Path(tmp) / AXONS_NAME))
            axon_network(restored, coords, config, Path(tmp) / "round_trip.png",
                         title="round_trip", seed=0)
            self.assertTrue((Path(tmp) / "round_trip.png").exists())


class TestModular4GridArea(unittest.TestCase):
    """`modular_4_grid` — 四隅の正方形を格子状に連結した領域 (areas.yaml に直書きした形)。

    `modular_4` が「隙間があるので非連結」なのに対し、こちらは parts が実際に重なって
    いることが存在意義なので、そこを固定する。

    **具体的な座標は書かない。** プロファイルは手で調整される前提なので、寸法を写すと
    値をいじるたびにテストが落ちる。代わりに parts から幾何を読み取って、寸法によらず
    成り立つべき性質だけを見る (パラメータから生成する版は TestModularGridArea)。
    """

    @staticmethod
    def _area():
        from src.models.network.area import Modular4GridArea
        return Modular4GridArea(_area_profile("modular_4_grid"))

    @staticmethod
    def _classify(area):
        """parts を「モジュール (正方形に近い)」と「ブリッジ (細長い)」に分ける。"""
        modules, bridges = [], []
        for part in area.parts:
            (x0, y0), (x1, y1) = part.bounds
            w, h = x1 - x0, y1 - y0
            (bridges if min(w, h) < 0.5 * max(w, h) else modules).append(
                (np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)]), np.array([x0, y0]), np.array([x1, y1])))
        return modules, bridges

    def test_registered_under_its_yaml_key(self):
        """profile_name は areas.yaml のキーであると同時にレジストリのキーでもある。"""
        from src.core.registry import AREA_MODELS
        from src.models.network.area import Modular4GridArea
        self.assertIs(AREA_MODELS.get("modular_4_grid"), Modular4GridArea)

    def test_is_a_single_connected_component(self):
        """格子として 1 つにつながっていること (modular_4 は 5 成分)。

        bbox に格子を敷いて連結成分を数える。格子間隔がブリッジ幅より粗いと細い通路が
        潰れて偽陽性になるので、最も細い part の幅から必要な解像度を決める。
        """
        from scipy import ndimage
        area = self._area()
        lo, hi = area.bounds
        thinnest = min(min(p.bounds[1] - p.bounds[0]) for p in area.parts)
        g = int(max(400, 4 * np.max(hi - lo) / thinnest))
        grid = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], g),
                                    np.linspace(lo[1], hi[1], g), indexing="ij"), -1)
        _, n_components = ndimage.label(area.contains(grid.reshape(-1, 2)).reshape(g, g))
        self.assertEqual(n_components, 1, f"{n_components} 個の成分に分かれている")

    def test_every_bridge_overlaps_a_module(self):
        """ブリッジは必ずモジュールに食い込んでいること (接するだけでは連結にならない)。

        `modular_4` が「40 um 届いていない」で 5 成分に割れたのと同じ失敗を防ぐ。
        """
        area = self._area()
        modules, bridges = self._classify(area)
        self.assertGreaterEqual(len(bridges), 1, "ブリッジらしき part がない")
        for _, lo, hi in bridges:
            # 矩形どうしの重なりは各軸の交差長がすべて正であること
            overlaps_a_module = any(
                ((np.minimum(hi, m_hi) - np.maximum(lo, m_lo)) > 0).all()
                for _, m_lo, m_hi in modules
            )
            self.assertTrue(overlaps_a_module,
                            f"ブリッジ {lo}..{hi} がどのモジュールとも重なっていない")

    def test_modules_are_connected_but_diagonals_are_not_direct(self):
        """隣接モジュールは直線で通れるが、対角は中央の空洞で遮られること。

        中心は parts から読むので、寸法を変えても成り立つ。
        """
        area = self._area()
        modules, _ = self._classify(area)
        self.assertEqual(len(modules), 4, "正方形モジュールが 4 つではない")
        centers = np.array([c for c, _, _ in modules])
        np.testing.assert_array_equal(area.contains(centers), [True] * 4)

        # 中心の x か y を共有する組 = 上下左右の隣。それ以外 = 対角。
        for i in range(4):
            for j in range(i + 1, 4):
                adjacent = np.isclose(centers[i], centers[j]).any()
                got = bool(area.segment_inside(centers[i:i+1], centers[j:j+1])[0])
                self.assertEqual(got, adjacent,
                                 f"{centers[i]} - {centers[j]}: 隣接={adjacent} なのに通過={got}")

    def test_axons_stay_inside_and_bridges_carry_traffic(self):
        """軸索が領域外へ出ず、かつモジュールをまたぐ結合が実際に生まれること。

        `modular_4` (非連結) では 0 でなければならないが、こちらは逆に
        **0 だとブリッジが機能していない**ことになる。
        """
        area = self._area()
        rng = np.random.RandomState(3)
        n = 800
        coords = np.zeros((n, 3), dtype=np.float32)
        coords[:, :2] = area.sample(n, rng)
        conn = AxonGrowthTopology(AXON_CFG, n, coords, rng, area=area)
        rows, cols = conn.generate_sparse()

        starts, ends, _, _ = conn._grow_axons(conn._params())
        self.assertTrue(area.segment_inside(starts, ends).all())

        # part index 0..3 が正方形、それ以降がブリッジ (areas.yaml の並び順)
        module = np.minimum(area.part_of(coords[:, :2]), 4)
        crossing = module[rows] != module[cols]
        self.assertGreater(int(crossing.sum()), 0, "ブリッジ経由の結合が 1 本もない")


class TestModularGridArea(unittest.TestCase):
    """`modular_grid` — 一辺 / ブリッジ幅 / 中心間距離 / モジュール数から自動生成する格子。

    直書きの `modular_4_grid` と違い形が計算で決まるので、**どのモジュール数でも
    「穴のない矩形格子」「連結成分は 1 つ」「斜めは直結しない」が成り立つこと**を見る。
    角は 45 度に面取りされる (正方形 ∩ 菱形)。
    """

    SIDE, BW, SPACING, CHAMFER = 500.0, 100.0, 750.0, 125.0   # 隙間 = 750 - 500 = 250

    @classmethod
    def _area(cls, **kwargs):
        from src.models.network.area import ModularGridArea
        params = dict(side=cls.SIDE, bridge_width=cls.BW, spacing=cls.SPACING, num_modules=4)
        params.update(kwargs)
        return ModularGridArea(_cfg(**params))

    @staticmethod
    def _n_components(area, g=800):
        from scipy import ndimage
        lo, hi = area.bounds
        grid = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], g),
                                    np.linspace(lo[1], hi[1], g), indexing="ij"), -1)
        _, n = ndimage.label(area.contains(grid.reshape(-1, 2)).reshape(g, g))
        return n

    def test_registered_and_yaml_profile_is_valid(self):
        """areas.yaml のプロファイルがそのまま構築できること。"""
        from src.core.registry import AREA_MODELS
        from src.models.network.area import ModularGridArea
        self.assertIs(AREA_MODELS.get("modular_grid"), ModularGridArea)
        area = ModularGridArea(_area_profile("modular_grid"))
        self.assertEqual((area.rows, area.cols), (2, 2))

    def test_grid_shape_is_the_squarest_exact_factorization(self):
        """num_modules だけから穴のない矩形格子を決める。素数は一列になる。"""
        for n, shape in ((1, (1, 1)), (2, (1, 2)), (4, (2, 2)), (5, (1, 5)),
                         (6, (2, 3)), (9, (3, 3)), (12, (3, 4)), (16, (4, 4))):
            with self.subTest(num_modules=n):
                area = self._area(num_modules=n)
                self.assertEqual((area.rows, area.cols), shape)
                self.assertEqual(len(area.parts), n + area.n_bridges)
                # 上下左右の隣にだけブリッジ: r*(c-1) + c*(r-1)
                r, c = shape
                self.assertEqual(area.n_bridges, r * (c - 1) + c * (r - 1))

    def test_always_a_single_connected_component(self):
        """既定の bridge_overlap で必ず連結になること (これが直書き版との違い)。"""
        for n in (1, 2, 3, 4, 6, 9):
            with self.subTest(num_modules=n):
                self.assertEqual(self._n_components(self._area(num_modules=n)), 1)

    def test_spacing_is_center_to_center(self):
        """spacing は**中心間距離**。辺と辺の隙間は spacing - side になる。"""
        area = self._area(num_modules=2)          # 1x2
        self.assertAlmostEqual(area.gap, self.SPACING - self.SIDE)
        centers = np.array([p.bounds.mean(axis=0) for p in area.parts[:2]])
        self.assertAlmostEqual(float(np.linalg.norm(centers[1] - centers[0])), self.SPACING)
        # side 以下の spacing はモジュールが重なるので受け付けない
        with self.assertRaises(ValueError):
            self._area(spacing=self.SIDE)

    def test_corners_are_chamfered_at_45_degrees(self):
        """角が 45 度に落ちていること。面取り面の外側は領域外、辺の直線部は内側。"""
        area = self._area(num_modules=1, chamfer=self.CHAMFER)
        h, c = self.SIDE / 2.0, self.CHAMFER
        eps = 1.0
        # 元の角 (h, h) は削られている。面取り面の内外を挟んで確認する。
        corner = np.array([[h - eps, h - eps]])
        self.assertFalse(area.contains(corner)[0], "角が削られていない")
        # 面取り面の始点 (h, h-c) 付近は辺の直線部なので内側
        np.testing.assert_array_equal(
            area.contains(np.array([[h - eps, h - c - eps], [h - c - eps, h - eps]])),
            [True, True])
        # 面取り面の中点はちょうど境界 (|x|+|y| = side - chamfer)
        mid = (self.SIDE - c) / 2.0
        self.assertAlmostEqual(float(area.sdf(np.array([[mid, mid]]))[0]), 0.0, places=6)
        # chamfer=0 なら素の正方形に戻る
        self.assertTrue(self._area(num_modules=1, chamfer=0.0).contains(corner)[0])

    def test_diagonal_modules_are_not_directly_connected(self):
        """ブリッジは上下左右だけ。2x2 の対角は直線で結べない (= 2 ホップ必要)。"""
        area = self._area(num_modules=4)
        centers = np.array([p.bounds.mean(axis=0) for p in area.parts[:4]])
        np.testing.assert_array_equal(area.contains(centers), [True] * 4)
        np.testing.assert_array_equal(
            area.segment_inside(centers[[0, 0, 0]], centers[[1, 2, 3]]),
            [True, True, False],   # 横・縦は通る、対角は通らない
        )

    def test_area_is_analytic_and_matches_the_grid_count(self):
        """解析面積 (密度表示に使う) が CompositeArea の格子カウントと一致すること。"""
        from src.models.network.area import CompositeArea
        area = self._area(num_modules=6, chamfer=self.CHAMFER)
        analytic = area.area_um2
        self.assertAlmostEqual(analytic / CompositeArea.area_um2.fget(area), 1.0, delta=0.01)
        # 内訳: 面取り済みモジュール 6 枚 (角 4 つ = 2*chamfer^2 を削る) + ブリッジ 7 本の露出部
        self.assertAlmostEqual(
            analytic,
            6 * (self.SIDE ** 2 - 2 * self.CHAMFER ** 2) + 7 * (self.SPACING - self.SIDE) * self.BW)

    def test_part_names_become_module_labels(self):
        """part_names が module 軸のラベルになるので、意味のある名前が付いていること。

        面取りありのモジュールは入れ子の composite になるが、part の数と名前は変わらない。
        """
        for chamfer in (0.0, self.CHAMFER):
            with self.subTest(chamfer=chamfer):
                area = self._area(num_modules=4, chamfer=chamfer)
                self.assertEqual(area.part_names[:4], ["M0", "M1", "M2", "M3"])
                self.assertTrue(all(n.startswith("B") for n in area.part_names[4:]))

    def test_generated_specs_survive_the_shared_helpers(self):
        """**回帰ガード**: 生成される part の座標を数値で固定する。

        モジュールとブリッジの spec は `_module_spec` / `_bridge_spec` に括り出されて
        `hierarchical_modular_grid` と共有されている。この式が 1 つでも変わると、
        座標が動く -> 棄却サンプリングのドロー数が変わる -> **同じ seed の既存
        ネットワークが重みと遅延まで丸ごと別物になる** (CLAUDE.md の注意 14)。
        面積・連結性・名前を見る他のテストはこの種のずれを通してしまうので、
        bbox そのものをここで押さえる。

        chamfer は bbox を変えない (菱形が正方形を内包する) ため、面取りの検証は
        `test_corners_are_chamfered_at_45_degrees` の担当。
        """
        expected = [
            ("M0",   (-625.0, -625.0, -125.0, -125.0)),
            ("M1",   ( 125.0, -625.0,  625.0, -125.0)),
            ("M2",   (-625.0,  125.0, -125.0,  625.0)),
            ("M3",   ( 125.0,  125.0,  625.0,  625.0)),
            ("B0-1", (-225.0, -425.0,  225.0, -325.0)),
            ("B0-2", (-425.0, -225.0, -325.0,  225.0)),
            ("B1-3", ( 325.0, -225.0,  425.0,  225.0)),
            ("B2-3", (-225.0,  325.0,  225.0,  425.0)),
        ]
        for chamfer in (0.0, self.CHAMFER):
            with self.subTest(chamfer=chamfer):
                area = self._area(num_modules=4, chamfer=chamfer)
                self.assertEqual(area.part_names, [n for n, _ in expected])
                for (name, box), part in zip(expected, area.parts):
                    np.testing.assert_allclose(
                        np.asarray(part.bounds).ravel(), box,
                        err_msg=f"{name} の位置が変わっている")

    def test_rejects_degenerate_parameters(self):
        for kwargs in (dict(side=-1.0), dict(bridge_width=0.0),
                       dict(spacing=400.0),           # 中心間距離が side 以下 = 重なる
                       dict(num_modules=0),
                       dict(chamfer=-1.0), dict(chamfer=250.0),   # side/2 で菱形に潰れる
                       dict(bridge_width=300.0, chamfer=125.0),   # 面取り面にはみ出す
                       dict(bridge_overlap=0.0)):     # 接するだけ = 連結にならない
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    self._area(**kwargs)

    def test_axons_stay_inside_and_bridges_carry_traffic(self):
        """生成された格子の上でも軸索は領域外へ出ず、ブリッジ経由の結合が生まれること。"""
        area = self._area(num_modules=6, side=600.0, bridge_width=120.0, spacing=850.0)
        rng = np.random.RandomState(5)
        coords = np.zeros((800, 3), dtype=np.float32)
        coords[:, :2] = area.sample(800, rng)
        conn = AxonGrowthTopology(AXON_CFG, 800, coords, rng, area=area)
        rows, cols = conn.generate_sparse()

        starts, ends, _, _ = conn._grow_axons(conn._params())
        self.assertTrue(area.segment_inside(starts, ends).all())

        module = np.minimum(area.part_of(coords[:, :2]), area.num_modules)  # 6 = ブリッジ
        self.assertGreater(int((module[rows] != module[cols]).sum()), 0,
                           "モジュールをまたぐ結合が 1 本もない")

    # ------------------------------------------- soma_in_bridge (ブリッジへの soma 配置)

    def test_soma_in_bridge_is_true_by_default(self):
        """キーを書かなければブリッジにも soma を置く = 従来どおりの実現。

        見るのは**クラスの既定値**であって areas.yaml の値ではない。プロファイルに何を
        書くかは実験ごとの選択なので、そちらを固定すると config を変えた瞬間にここが落ちる。
        プロファイルが構築できること自体は
        `test_registered_and_yaml_profile_is_valid` が見ている。
        """
        area = self._area(num_modules=4)                 # soma_in_bridge を渡さない
        self.assertTrue(area.soma_in_bridge)
        self.assertTrue(all(area.part_allows_soma))
        self.assertIs(area.soma_area, area)              # 乱数消費が変わらないことの担保

    def test_soma_in_bridge_false_keeps_somas_out_of_bridges(self):
        """soma はモジュール内だけに落ち、module 軸から B* が消えること。"""
        area = self._area(num_modules=6, soma_in_bridge=False)
        self.assertEqual(area.part_allows_soma,
                         [True] * area.num_modules + [False] * area.n_bridges)

        region = area.soma_area
        self.assertEqual(region.part_names, [f"M{i}" for i in range(area.num_modules)])

        pts = region.sample(2000, np.random.RandomState(3))
        self.assertTrue(area.contains(pts).all(), "soma が領域外に出た")
        # どの点もいずれかのモジュール part の内側 = ブリッジの露出部には 1 点も無い
        in_module = np.zeros(len(pts), dtype=bool)
        for part in area.parts[:area.num_modules]:
            in_module |= part.contains(pts)
        self.assertTrue(in_module.all(), "soma がブリッジに落ちた")
        # module 軸のラベル。食い込み帯でも B* にならないこと (全体領域で part_of を
        # 採ると「より深い」ブリッジ part が選ばれてしまうのを防いでいる)
        labels = np.asarray(region.part_names)[region.part_of(pts)]
        self.assertTrue(all(str(n).startswith("M") for n in labels))

    def test_soma_area_um2_counts_modules_only(self):
        """密度の分母になる soma 配置領域の面積は、ブリッジの露出分を除いた解析値。"""
        from src.models.network.area import CompositeArea
        area = self._area(num_modules=6, chamfer=self.CHAMFER, soma_in_bridge=False)
        expected = 6 * (self.SIDE ** 2 - 2 * self.CHAMFER ** 2)
        self.assertAlmostEqual(area.soma_area.area_um2, expected)
        self.assertAlmostEqual(area.area_um2 - expected,
                               area.n_bridges * (self.SPACING - self.SIDE) * self.BW)
        # 格子カウントとも整合していること (解析値が形と食い違っていない)
        self.assertAlmostEqual(
            expected / CompositeArea.area_um2.fget(area.soma_area), 1.0, delta=0.01)

    def test_axons_still_cross_bridges_when_somas_are_excluded(self):
        """**要件の本体**: soma をブリッジから外しても軸索はブリッジを通って結合を作る。"""
        area = self._area(num_modules=6, side=600.0, bridge_width=120.0, spacing=850.0,
                          soma_in_bridge=False)
        rng = np.random.RandomState(5)
        coords = np.zeros((800, 3), dtype=np.float32)
        coords[:, :2] = area.soma_area.sample(800, rng)
        conn = AxonGrowthTopology(AXON_CFG, 800, coords, rng, area=area)
        rows, cols = conn.generate_sparse()

        starts, ends, _, _ = conn._grow_axons(conn._params())
        self.assertTrue(area.segment_inside(starts, ends).all())

        module = area.soma_area.part_of(coords[:, :2])
        self.assertGreater(int((module[rows] != module[cols]).sum()), 0,
                           "モジュールをまたぐ結合が 1 本もない")

    def test_area_uniform_space_uses_the_soma_region(self):
        """`AreaUniformSpace` が soma 配置領域からサンプルし、module 軸を M* だけで出すこと。"""
        area = self._area(num_modules=4, soma_in_bridge=False)
        space = AreaUniformSpace(_cfg(), 300, np.random.RandomState(7), area=area)
        coords = space.generate()

        labels = space.describe_axes()["module"]
        self.assertEqual(len(labels), 300)
        self.assertTrue(all(str(n).startswith("M") for n in labels))
        self.assertTrue(area.contains(coords[:, :2]).all())
        # 軸索側へ渡るのは領域そのもの (space が area を差し替えたりしない)
        self.assertIs(space.area, area)

    # ---------------------------------------- module 軸 (所属モジュールのラベル割り当て)

    def test_module_axis_labels_match_the_geometric_module(self):
        """`layout.ids_by("module")["M{i}"]` の ID が実際に i 番目のモジュール内にあること。

        この軸の意味は「そのニューロンがどのモジュールに置かれたか」なので、カテゴリ名が
        揃っているだけでは足りない。**ラベルと幾何の一致**まで見る。
        """
        from src.core.layout import NetworkLayout

        n = 400
        area = self._area(num_modules=6, soma_in_bridge=False)
        space = AreaUniformSpace(_cfg(), n, np.random.RandomState(11), area=area)
        coords = space.generate()

        # NetworkBuilder._inject_axes() と同じことを手で行う (config 解決は要らない)
        layout = NetworkLayout(["Layer_Exc"] * n)
        for name, values in space.describe_axes().items():
            layout.add_axis(name, values)

        ids = layout.ids_by("module")
        self.assertEqual(sorted(ids), [f"M{i}" for i in range(area.num_modules)])
        self.assertEqual(sum(len(v) for v in ids.values()), n)
        for i in range(area.num_modules):
            with self.subTest(module=i):
                xy = coords[ids[f"M{i}"], :2]
                self.assertTrue(area.parts[i].contains(xy).all(),
                                f"M{i} のラベルが付いた soma が {i} 番目のモジュール外にある")

    def test_module_axis_includes_bridges_when_somas_are_allowed_there(self):
        """既定 (soma_in_bridge 省略) ではブリッジも独自カテゴリとして module 軸に出る。

        「所属モジュール」を素直に得たいときの落とし穴なので、**仕様として固定**しておく。
        M0..M{n-1} だけにしたいなら soma_in_bridge: false
        (`test_area_uniform_space_uses_the_soma_region` がそちらを見ている)。
        """
        area = self._area(num_modules=4)
        self.assertIs(area.soma_area, area)   # 既定なので soma 配置領域は領域そのもの

        space = AreaUniformSpace(_cfg(), 600, np.random.RandomState(13), area=area)
        space.generate()

        labels = set(map(str, space.describe_axes()["module"]))
        self.assertTrue(labels <= set(area.part_names))
        self.assertTrue(any(n.startswith("B") for n in labels),
                        "ブリッジに soma が 1 つも落ちていない (この config では起きないはず)")


class TestHierarchicalModularGridArea(unittest.TestCase):
    """`hierarchical_modular_grid` — 2x2 クラスタを 2x2 に並べた **2 階層**のモジュール格子。

    `modular_grid` との違いは「どのモジュール同士を繋ぐか」だけ。クラスタ内は 4-サイクル、
    クラスタ間は**内側 1 モジュール同士**だけを繋ぐので、4x4 格子の全直交辺 24 本から
    境界の外側 4 辺を抜いた形になる。ここで見るのはその選択が実際に効いていること —
    内側でない向かい合わせが繋がっていないこと (`modular_grid(16)` との差) が中心。
    """

    SIDE, BW, SPACING, BLOCK, CHAMFER = 500.0, 100.0, 750.0, 1600.0, 125.0

    # 上のパラメータでのモジュール中心。クラスタ中心 (±800, ±800) にモジュールの
    # オフセット (±375, ±375) を足したもの。**式ではなく値で持つ** — 実装と同じ式で
    # 期待値を作ると、式ごと間違えたときに素通りする。
    CENTERS = {
        "C0-M0": (-1175.0, -1175.0), "C0-M1": (-425.0, -1175.0),
        "C0-M2": (-1175.0,  -425.0), "C0-M3": (-425.0,  -425.0),
        "C1-M0": (  425.0, -1175.0), "C1-M1": (1175.0, -1175.0),
        "C1-M2": (  425.0,  -425.0), "C1-M3": (1175.0,  -425.0),
        "C2-M0": (-1175.0,   425.0), "C2-M1": (-425.0,   425.0),
        "C2-M2": (-1175.0,  1175.0), "C2-M3": (-425.0,  1175.0),
        "C3-M0": (  425.0,   425.0), "C3-M1": (1175.0,   425.0),
        "C3-M2": (  425.0,  1175.0), "C3-M3": (1175.0,  1175.0),
    }
    # 各クラスタで全体の中心に最も近いモジュール = クラスタ間ブリッジが生える場所
    INNER = ("C0-M3", "C1-M2", "C2-M1", "C3-M0")
    # クラスタ間ブリッジで直結している内側モジュール対 (格子の直交隣接)
    INNER_PAIRS = (("C0-M3", "C1-M2"), ("C2-M1", "C3-M0"),
                   ("C0-M3", "C2-M1"), ("C1-M2", "C3-M0"))

    @classmethod
    def _area(cls, **kwargs):
        from src.models.network.area import HierarchicalModularGridArea
        params = dict(side=cls.SIDE, bridge_width=cls.BW,
                      spacing=cls.SPACING, block_spacing=cls.BLOCK)
        params.update(kwargs)
        return HierarchicalModularGridArea(_cfg(**params))

    @staticmethod
    def _centers(area):
        """part 名 -> 中心座標。モジュールは矩形か「矩形 ∩ 菱形」なので bbox 中心でよい。"""
        return {n: tuple(np.asarray(p.bounds).mean(axis=0))
                for n, p in zip(area.part_names, area.parts)}

    def _seg(self, area, a, b):
        """モジュール a の中心から b の中心まで、領域内だけを通って行けるか。"""
        c = self._centers(area)
        return bool(area.segment_inside(np.array([c[a]]), np.array([c[b]]))[0])

    # ------------------------------------------------------------------ 構成

    def test_registered_and_yaml_profile_is_valid(self):
        """areas.yaml のプロファイルがそのまま構築でき、4 クラスタ x 4 モジュールであること。"""
        from src.core.registry import AREA_MODELS
        from src.models.network.area import HierarchicalModularGridArea
        self.assertIs(AREA_MODELS.get("hierarchical_modular_grid"), HierarchicalModularGridArea)
        area = HierarchicalModularGridArea(_area_profile("hierarchical_modular_grid"))
        self.assertEqual((area.num_clusters, area.modules_per_cluster), (4, 4))
        self.assertEqual(area.num_modules, 16)

    def test_sixteen_modules_then_intra_then_inter_bridges(self):
        """part の構成と並び順。名前がそのまま module 軸になるので名前まで固定する。"""
        area = self._area()
        self.assertEqual((area.num_modules, area.n_intra_bridges, area.n_inter_bridges),
                         (16, 16, 4))
        self.assertEqual(len(area.parts), 36)

        # 1. モジュール 16 枚 (クラスタが外側、モジュールが内側)
        self.assertEqual(area.part_names[:16],
                         [f"C{c}-M{m}" for c in range(4) for m in range(4)])
        # 2. クラスタ内ブリッジ 16 本 (各クラスタ 4 本の 4-サイクル)
        self.assertEqual(area.part_names[16:32],
                         [f"BC{c}-{a}-{b}" for c in range(4)
                          for a, b in ((0, 1), (0, 2), (1, 3), (2, 3))])
        # 3. クラスタ間ブリッジ 4 本。クラスタ対 1 つにつき 1 本なので対で一意
        self.assertEqual(area.part_names[32:], ["BX0-1", "BX0-2", "BX1-3", "BX2-3"])

    def test_module_names_sort_into_cluster_blocks(self):
        """`C{c}-M{m}` 形式は**文字列ソートでクラスタごとのブロックになる**。

        これが `order_by("module")` や `group_connection_probability` が追加の軸なしで
        階層を見せられる理由なので、命名規則として固定しておく。
        """
        names = self._area().part_names[:16]
        self.assertEqual(sorted(names), names)
        for c in range(4):
            block = [i for i, n in enumerate(sorted(names)) if n.startswith(f"C{c}-")]
            self.assertEqual(block, list(range(4 * c, 4 * c + 4)))

    def test_module_centers_are_a_2x2_of_2x2(self):
        """クラスタ中心に spacing の 2x2、クラスタ同士は block_spacing の 2x2。"""
        centers = self._centers(self._area())
        for name, expected in self.CENTERS.items():
            with self.subTest(module=name):
                np.testing.assert_allclose(centers[name], expected)
        # 内側 4 モジュールが本当に原点に最も近いこと (INNER の定義そのもの)
        by_dist = sorted(self.CENTERS, key=lambda n: np.hypot(*self.CENTERS[n]))
        self.assertEqual(set(by_dist[:4]), set(self.INNER))

    # ------------------------------------------------------------ 繋がり方 (要件の中核)

    def test_only_the_inner_modules_bridge_between_clusters(self):
        """**要件の中核**: クラスタ間の通り道は内側モジュール対の 4 本だけ。

        `modular_grid(16)` との差はここにしかない。内側同士の直交 4 対が通れて、
        **内側でない向かい合わせは通れない**ことまで見ないと両者を区別できない。
        """
        area = self._area()
        for a, b in self.INNER_PAIRS:
            with self.subTest(pair=(a, b), expect=True):
                self.assertTrue(self._seg(area, a, b), f"{a}-{b} が繋がっていない")
        # 内側同士でも斜向かいは直結しない (2 ホップ)
        for a, b in (("C0-M3", "C3-M0"), ("C1-M2", "C2-M1")):
            with self.subTest(pair=(a, b), expect=False):
                self.assertFalse(self._seg(area, a, b), f"{a}-{b} が直結している")
        # **内側でない向かい合わせ**は繋がらない。ここが「全ペア接続」との分かれ目で、
        # 4x4 格子から抜いた 4 辺にあたる。
        for a, b in (("C0-M1", "C1-M0"), ("C2-M3", "C3-M2"),
                     ("C0-M2", "C2-M0"), ("C1-M3", "C3-M1")):
            with self.subTest(pair=(a, b), expect=False):
                self.assertFalse(self._seg(area, a, b),
                                 f"{a}-{b} が繋がっている (内側だけを繋ぐ約束が壊れている)")

    def test_intra_cluster_is_a_four_cycle(self):
        """クラスタ内は modular_grid(4) と同じ 4-サイクル。対角は 2 ホップ。"""
        area = self._area()
        for c in range(4):
            for a, b in ((0, 1), (0, 2), (1, 3), (2, 3)):
                with self.subTest(cluster=c, pair=(a, b), expect=True):
                    self.assertTrue(self._seg(area, f"C{c}-M{a}", f"C{c}-M{b}"))
            for a, b in ((0, 3), (1, 2)):
                with self.subTest(cluster=c, pair=(a, b), expect=False):
                    self.assertFalse(self._seg(area, f"C{c}-M{a}", f"C{c}-M{b}"))

    def test_always_a_single_connected_component(self):
        """20 本のブリッジで全体が 1 つに繋がること (孤立クラスタを作らない)。"""
        self.assertEqual(TestModularGridArea._n_components(self._area()), 1)

    # ------------------------------------------------------------------ block_spacing

    def test_default_block_spacing_gives_a_uniform_grid(self):
        """`block_spacing` 既定 = 2*spacing は「均一な 4x4 格子」になる。

        このとき**形**は modular_grid(16) と同じで、違うのは張られるブリッジの数だけ
        (24 本 -> 20 本)。階層をはっきりさせたいなら block_spacing を上げる、の根拠。
        """
        from src.models.network.area import HierarchicalModularGridArea
        area = HierarchicalModularGridArea(
            _cfg(side=self.SIDE, bridge_width=self.BW, spacing=self.SPACING))
        self.assertAlmostEqual(area.block_spacing, 2.0 * self.SPACING)
        # クラスタ間の隙間がクラスタ内と等しい = 均一
        self.assertAlmostEqual(area.inter_gap, area.gap)
        xs = np.unique(np.round([c[0] for c in self._centers(area).values()][:16], 6))
        self.assertEqual(len(xs), 4)
        np.testing.assert_allclose(np.diff(xs), self.SPACING)

    def test_larger_block_spacing_only_stretches_the_cluster_gap(self):
        """block_spacing を上げるとクラスタ間ブリッジだけが伸び、クラスタ内は不変。"""
        near, far = self._area(block_spacing=1600.0), self._area(block_spacing=2400.0)
        self.assertAlmostEqual(near.gap, far.gap)                       # クラスタ内は同じ
        self.assertAlmostEqual(near.inter_gap, 1600.0 - 750.0 - 500.0)  # = 350
        self.assertAlmostEqual(far.inter_gap, 2400.0 - 750.0 - 500.0)   # = 1150
        # クラスタ内のモジュール中心間距離は spacing のまま
        for area in (near, far):
            c = self._centers(area)
            self.assertAlmostEqual(
                float(np.linalg.norm(np.subtract(c["C0-M1"], c["C0-M0"]))), self.SPACING)

    # ------------------------------------------------------------------ 面積 / soma

    def test_area_is_analytic_and_matches_the_grid_count(self):
        """解析面積 (密度表示の分母) が CompositeArea の格子カウントと一致すること。"""
        from src.models.network.area import CompositeArea
        area = self._area(chamfer=self.CHAMFER)
        analytic = area.area_um2
        self.assertAlmostEqual(analytic / CompositeArea.area_um2.fget(area), 1.0, delta=0.01)
        # 内訳: 面取り済みモジュール 16 枚 + 露出長の違う 2 種類のブリッジ
        self.assertAlmostEqual(
            analytic,
            16 * (self.SIDE ** 2 - 2 * self.CHAMFER ** 2)
            + 16 * (self.SPACING - self.SIDE) * self.BW
            + 4 * (self.BLOCK - self.SPACING - self.SIDE) * self.BW)

    def test_soma_in_bridge_false_keeps_somas_out_of_bridges(self):
        """soma はモジュール内だけに落ち、module 軸が 16 個の C*-M* だけになること。"""
        from src.models.network.area import CompositeArea
        area = self._area(chamfer=self.CHAMFER, soma_in_bridge=False)
        self.assertEqual(area.part_allows_soma, [True] * 16 + [False] * 20)

        region = area.soma_area
        self.assertEqual(region.part_names, [f"C{c}-M{m}" for c in range(4) for m in range(4)])

        expected = 16 * (self.SIDE ** 2 - 2 * self.CHAMFER ** 2)
        self.assertAlmostEqual(region.area_um2, expected)
        self.assertAlmostEqual(expected / CompositeArea.area_um2.fget(region), 1.0, delta=0.01)

        pts = region.sample(2000, np.random.RandomState(3))
        self.assertTrue(area.contains(pts).all(), "soma が領域外に出た")
        # どの点もいずれかのモジュール part の内側 = ブリッジの露出部には 1 点も無い
        in_module = np.zeros(len(pts), dtype=bool)
        for part in area.parts[:16]:
            in_module |= part.contains(pts)
        self.assertTrue(in_module.all(), "soma がブリッジに落ちた")
        labels = np.asarray(region.part_names)[region.part_of(pts)]
        self.assertTrue(all(str(n).startswith("C") for n in labels))

    # ------------------------------------------------------------------ 退化パラメータ

    def test_rejects_degenerate_parameters(self):
        for kwargs in (dict(side=-1.0), dict(bridge_width=0.0),
                       dict(spacing=400.0),           # 中心間距離が side 以下 = 重なる
                       dict(chamfer=-1.0), dict(chamfer=250.0),   # side/2 で菱形に潰れる
                       dict(bridge_width=300.0, chamfer=125.0),   # 面取り面にはみ出す
                       dict(bridge_overlap=0.0),      # 接するだけ = 連結にならない
                       dict(block_spacing=1250.0),    # = spacing + side、内側同士が接する
                       dict(block_spacing=1000.0)):   # < spacing + side、内側同士が重なる
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    self._area(**kwargs)

    def test_rejects_grid_shape_keys(self):
        """4x4 固定なので num_modules / rows / cols は黙って無視せずエラーにする。

        受け取って無視すると「16 個指定したのに 16 個できている」ように見えてしまい、
        `modular_grid` と取り違えたことに気づけない。
        """
        for key in ("num_modules", "rows", "cols"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as cm:
                    self._area(**{key: 4})
                self.assertIn("modular_grid", str(cm.exception))   # 代替の案内が出ること

    # ------------------------------------------------------------------ 軸索と module 軸

    def test_axons_cross_both_bridge_levels(self):
        """**要件の本体**: soma をブリッジから外しても軸索は両方のブリッジを通る。

        クラスタ内 (モジュールをまたぐ) とクラスタ間の結合が両方生まれ、かつ
        **クラスタ間の結合が内側モジュールに集中する**ことまで見る。後者が出ないなら
        軸索がブリッジ以外の場所を通れてしまっている。
        """
        area = self._area(side=600.0, bridge_width=120.0, spacing=850.0,
                          block_spacing=1700.0, soma_in_bridge=False)
        n = 600
        rng = np.random.RandomState(5)
        coords = np.zeros((n, 3), dtype=np.float32)
        coords[:, :2] = area.soma_area.sample(n, rng)
        conn = AxonGrowthTopology(AXON_CFG, n, coords, rng, area=area)
        rows, cols = conn.generate_sparse()

        starts, ends, _, _ = conn._grow_axons(conn._params())
        self.assertTrue(area.segment_inside(starts, ends).all(), "軸索が領域外へ出た")

        labels = np.asarray(area.soma_area.part_names)[area.soma_area.part_of(coords[:, :2])]
        cluster = np.array([s.split("-")[0] for s in labels])
        cross_cluster = cluster[rows] != cluster[cols]
        cross_module = (labels[rows] != labels[cols]) & ~cross_cluster

        self.assertGreater(int(cross_module.sum()), 0, "クラスタ内でモジュールをまたぐ結合が無い")
        self.assertGreater(int(cross_cluster.sum()), 0, "クラスタをまたぐ結合が無い")

        # クラスタを出る道は内側モジュールのブリッジしかないので、クラスタ間の結合は
        # **内側モジュールの近くに集中する**。ただし 100% ではない — 軸索は
        # mean_axon_length = 1100 um あるので、ブリッジを渡った先でさらに隣の
        # モジュールまで届くことがある (この config は spacing = 850 um なので稀だが、
        # spacing を詰めると 1 割ほど出る)。**構造的な保証ではないので比率で見る。**
        # 16 モジュール中 4 つが内側なので、端点が無作為なら 1 - (12/16)^2 = 44%。
        # それを大きく上回ることが「ブリッジ以外を通っていない」ことの証拠になる。
        inner = set(self.INNER)
        touches_inner = np.array([(a in inner) or (b in inner)
                                  for a, b in zip(labels[rows][cross_cluster],
                                                  labels[cols][cross_cluster])])
        self.assertGreater(
            touches_inner.mean(), 0.75,
            "クラスタ間結合が内側モジュールに集中していない (無作為なら 44%) = "
            "ブリッジ以外の場所を通れてしまっている")

    def test_module_axis_labels_match_the_geometric_module(self):
        """`layout.ids_by("module")["C1-M2"]` の ID が実際にそのモジュール内にあること。

        この軸の意味は「どのモジュールに置かれたか」なので、カテゴリ名が揃っている
        だけでは足りない。**ラベルと幾何の一致**まで見る。
        """
        from src.core.layout import NetworkLayout

        n = 500
        area = self._area(soma_in_bridge=False)
        space = AreaUniformSpace(_cfg(), n, np.random.RandomState(11), area=area)
        coords = space.generate()

        # NetworkBuilder._inject_axes() と同じことを手で行う (config 解決は要らない)
        layout = NetworkLayout(["Layer_Exc"] * n)
        for name, values in space.describe_axes().items():
            layout.add_axis(name, values)

        ids = layout.ids_by("module")
        self.assertEqual(sorted(ids), [f"C{c}-M{m}" for c in range(4) for m in range(4)])
        self.assertEqual(sum(len(v) for v in ids.values()), n)
        for i, name in enumerate(area.part_names[:16]):
            with self.subTest(module=name):
                xy = coords[ids[name], :2]
                self.assertTrue(area.parts[i].contains(xy).all(),
                                f"{name} のラベルが付いた soma がそのモジュール外にある")


class TestDiamondArea(unittest.TestCase):
    """`diamond` — 45 度の面取りを作るための菱形プリミティブ。"""

    @staticmethod
    def _area(**kw):
        from src.models.network.area import DiamondArea
        return DiamondArea(_cfg(**{"radius": 1.0, **kw}))

    def test_sdf_is_the_exact_euclidean_distance(self):
        """辺に面した側も頂点の外側も真の距離を返すこと (法線が正しく取れる条件)。"""
        area = self._area()
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 0.5], [3.0, 4.0]])
        expected = [-np.sqrt(0.5),            # 中心 -> 辺
                    0.0,                      # 頂点 (境界上)
                    1.0,                      # 頂点の外 1.0
                    0.0,                      # 辺の中点 (境界上)
                    np.hypot(3.0, 3.0)]       # 頂点 (0,1) までの距離
        np.testing.assert_allclose(area.sdf(pts), expected, atol=1e-9)

    def test_anisotropic_radius_and_area(self):
        area = self._area(radius=[3.0, 2.0])
        self.assertAlmostEqual(area.area_um2, 2 * 3.0 * 2.0)     # 対角線の積 / 2
        np.testing.assert_allclose(area.bounds, [[-3.0, -2.0], [3.0, 2.0]])
        np.testing.assert_array_equal(
            area.contains(np.array([[2.9, 0.0], [0.0, 2.1], [1.5, 1.0]])), [True, False, True])

    def test_convex_segment_inside_matches_sampling(self):
        """凸なので端点判定に短絡してよい。基底のサンプリング実装と一致すること。"""
        area = self._area(radius=[3.0, 2.0], center=[1.0, -1.0])
        rng = np.random.RandomState(1)
        a = rng.uniform(-5.0, 5.0, size=(400, 2))
        b = rng.uniform(-5.0, 5.0, size=(400, 2))
        np.testing.assert_array_equal(area.segment_inside(a, b),
                                      BaseArea.segment_inside(area, a, b))

    def test_rejects_non_positive_radius(self):
        with self.assertRaises(ValueError):
            self._area(radius=0.0)


if __name__ == "__main__":
    unittest.main()
