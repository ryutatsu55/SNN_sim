"""エリア可視化 (src/utils/plotting/area.py) の回帰テスト。

SDF を格子上で評価して描くやり方には、実装時に踏んだ罠が 3 つある。ここで固定するのは
その 3 つと、描画が乱数に触らないこと。
"""

import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")          # 表示のない環境で走らせる (import より前に設定)
import matplotlib.pyplot as plt
import numpy as np

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.models.network.area import DiskArea, Modular4Area, NoSpaceArea, RectArea
from src.utils.plotting.area import DEFAULT_PAD, draw_area, plot_area


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


class TestDrawArea(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_bounded_area_is_drawn(self):
        fig, ax = plt.subplots()
        self.assertTrue(draw_area(ax, DiskArea(_cfg(radius=1500.0))))
        self.assertTrue(ax.collections, "塗りも境界線も描かれていない")
        self.assertEqual(ax.get_aspect(), 1.0, "等方スケールになっていない")

    def test_unbounded_area_is_skipped(self):
        """罠 3: no_space は bounds=±inf / sdf=-inf なので描きようがない。

        判定は has_space (座標の有無) ではなく area.is_bounded で行う。両者は独立で、
        有界なエリアは no_space の座標とも共存しうる。
        """
        fig, ax = plt.subplots()
        self.assertFalse(draw_area(ax, NoSpaceArea(_cfg())))
        self.assertEqual(len(ax.collections), 0, "無界なのに何か描かれている")
        self.assertFalse(plot_area(NoSpaceArea(_cfg()), Path("/dev/null/never")))

    def test_none_area_is_skipped(self):
        fig, ax = plt.subplots()
        self.assertFalse(draw_area(ax, None))
        self.assertEqual(len(ax.collections), 0)

    def test_padding_keeps_the_boundary_off_the_grid_edge(self):
        """罠 1: パディングは必須。

        rect は領域が境界箱いっぱいに広がるので、pad=0 だと sdf=0 の等高線がちょうど
        格子の端に載り、contour が閉じない。軸範囲が bounds より広いことで、格子が
        領域より外まで張られている = 境界線の周囲に余白があることを担保する。
        """
        area = RectArea(_cfg(x_range=[0.0, 1500.0], y_range=[0.0, 1500.0]))
        lo, hi = area.bounds

        fig, ax = plt.subplots()
        draw_area(ax, area)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        self.assertLess(x0, lo[0], "左に余白が無い (pad=0 相当)")
        self.assertGreater(x1, hi[0], "右に余白が無い (pad=0 相当)")
        self.assertLess(y0, lo[1])
        self.assertGreater(y1, hi[1])
        self.assertGreater(DEFAULT_PAD, 0.0, "DEFAULT_PAD を 0 にしてはいけない")

    def test_part_fill_is_masked_to_the_region(self):
        """罠 2: part_of() は領域外の点にも index を返す。

        contains() でマスクしないと、argmin が必ず何かの part を指すせいで境界箱全体が
        塗られてしまう。境界箱の角 (領域の外) が塗られた領域に含まれないことを見る。
        """
        area = Modular4Area(MODULAR_SPEC)
        lo, hi = area.bounds
        corner = np.array([[hi[0], hi[1]]])
        # 前提: 角は確かに領域外だが、part_of は part を返す (これが罠の正体)
        self.assertFalse(bool(area.contains(corner)[0]))
        self.assertGreaterEqual(int(area.part_of(corner)[0]), 0)

        fig, ax = plt.subplots()
        draw_area(ax, area, by_part=True)
        for collection in ax.collections:
            for path in collection.get_paths():
                self.assertFalse(
                    path.contains_point((corner[0, 0], corner[0, 1])),
                    "領域外の角が塗られている (contains によるマスクが抜けている)",
                )

    def test_part_labels_do_not_collide(self):
        """重心が一致する part (十字の縦棒と横棒) でもラベルが重ならないこと。"""
        fig, ax = plt.subplots()
        draw_area(ax, Modular4Area(MODULAR_SPEC), by_part=True)
        positions = [tuple(t.get_position()) for t in ax.texts]
        self.assertEqual(len(positions), 6, "part の数だけラベルが出ていない")
        self.assertEqual(len(set(positions)), 6, "ラベルの位置が重複している")

    def test_fill_false_draws_boundary_only(self):
        """network() が使う経路。塗りを外しても境界線は残る。"""
        fig, ax = plt.subplots()
        self.assertTrue(draw_area(ax, Modular4Area(MODULAR_SPEC), fill=False))
        self.assertTrue(ax.collections, "境界線が描かれていない")
        self.assertEqual(len(ax.texts), 0, "塗らないのに part ラベルが出ている")

    def test_drawing_consumes_no_randomness(self):
        """描画は RandomState に触らない (area_um2 も決定論的な格子カウント)。"""
        rng = np.random.RandomState(3)
        before = rng.get_state()
        fig, ax = plt.subplots()
        draw_area(ax, Modular4Area(MODULAR_SPEC))
        after = rng.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


class TestPlotArea(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_writes_a_png(self):
        import tempfile

        area = DiskArea(_cfg(radius=1500.0))
        area.num_neurons = 2827
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sub" / "area.png"
            self.assertTrue(plot_area(area, out, title="Area: disk"))
            self.assertTrue(out.exists(), "親ディレクトリごと作られていない")
            self.assertGreater(out.stat().st_size, 0)


class TestAxonNetwork(unittest.TestCase):
    """`axon_network()` — 結合を軸索の折れ線で描く図。

    見た目は検証できないので、**何本の折れ線をどの頂点で描いたか**を LineCollection への
    引数として捕まえて確かめる。特に大事なのは、`network()` と同じ seed なら同じ
    ニューロンが描かれること (2 枚を並べて比較する前提が崩れないこと)。
    """

    N = 200
    SEED = 3

    def tearDown(self):
        plt.close("all")

    @classmethod
    def _build(cls, n=None):
        from src.models.network.connectors import AxonGrowthTopology

        n = n or cls.N
        area = DiskArea(_cfg(radius=700.0))
        rng = np.random.RandomState(cls.SEED)
        coords = np.zeros((n, 3), dtype=np.float64)
        coords[:, :2] = area.sample(n, rng)
        cfg = _cfg(mean_axon_length=1100.0, segment_length=100.0, angle_sigma=0.1,
                   dendrite_radius=150.0, connection_prob=0.2, boundary="deflect",
                   allow_self_connections=False)
        conn = AxonGrowthTopology(cfg, n, coords, rng, area=area)
        conn.generate_sparse()
        return area, conn.axon_geometry(), coords

    @staticmethod
    def _config():
        """`axon_network` が触るのは config.network.space だけ (エリアを渡せば見もしない)。"""
        return _cfg(network=_cfg(space=_cfg()))

    def _draw(self, tmp, **kwargs):
        """描画して、LineCollection に渡された折れ線と、描かれたニューロンを回収する。"""
        from unittest import mock

        # `import src.utils.plotting.network` ではモジュールを掴めない。パッケージの
        # __init__ が同名の関数 network を re-export しているので、属性参照が関数に
        # 解決されてしまう。sys.modules を引く import_module なら確実にモジュール。
        netmod = importlib.import_module("src.utils.plotting.network")

        area, geometry, coords = self._build()
        collections, sampled = [], []
        real_lc, real_nodes = netmod.LineCollection, netmod._draw_nodes

        def spy_lc(segments, **kw):
            collections.append([np.asarray(v) for v in segments])
            return real_lc(segments, **kw)

        def spy_nodes(ax, x, y, sample, is_exc, node_size):
            sampled.append(np.asarray(sample))
            return real_nodes(ax, x, y, sample, is_exc, node_size)

        with mock.patch.object(netmod, "LineCollection", spy_lc), \
                mock.patch.object(netmod, "_draw_nodes", spy_nodes):
            netmod.axon_network(geometry, coords, self._config(), area=area,
                                save_path=str(tmp), seed=self.SEED, **kwargs)
        return area, geometry, coords, collections, sampled[0]

    def test_writes_a_png_and_draws_three_layers(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, collections, _ = self._draw(Path(tmp), title="axon_network")
            self.assertTrue((Path(tmp) / "axon_network.png").exists())
            # 下敷き (全軸索) / 結合経路 / 樹状突起の破線 の 3 枚
            self.assertEqual(len(collections), 3)

    def test_paths_run_from_the_presynaptic_soma_to_the_contact_point(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _, g, coords, collections, sample = self._draw(Path(tmp))
        _, paths, stubs = collections

        in_sample = np.zeros(len(coords), dtype=bool)
        in_sample[sample] = True
        expected = np.nonzero(in_sample[g.pre] & in_sample[g.post])[0]
        self.assertEqual(len(paths), expected.size, "描いた結合数が選んだ結合数と違う")
        self.assertEqual(len(stubs), expected.size)

        for edge, path, stub in zip(expected, paths, stubs):
            pre, post = int(g.pre[edge]), int(g.post[edge])
            seg = int(g.contact_seg[edge])
            # 始点は pre の細胞体、終点は接触点、頂点数は経路長 + 接触点の 1 つ
            np.testing.assert_allclose(path[0], coords[pre, :2], atol=1e-6)
            self.assertEqual(len(path), seg - int(g.offsets[pre]) + 2)
            np.testing.assert_allclose(stub[0], path[-1], atol=1e-9)
            np.testing.assert_allclose(stub[1], coords[post, :2], atol=1e-6)

    def test_samples_the_same_neurons_as_network(self):
        """2 枚を並べて比較できることの担保。同じ seed なら同じニューロンが出る。"""
        import tempfile

        from unittest import mock

        netmod = importlib.import_module("src.utils.plotting.network")

        with tempfile.TemporaryDirectory() as tmp:
            _, g, coords, _, axon_sample = self._draw(Path(tmp))

            weights = np.ones(len(g.pre))
            with mock.patch.object(netmod, "_draw_nodes") as spy:
                netmod.network(g.pre, g.post, weights, coords, config=self._config(),
                               title="network_sample", save_path=str(tmp), seed=self.SEED)
        np.testing.assert_array_equal(spy.call_args[0][3], axon_sample)

    def test_underlay_can_be_turned_off(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, collections, _ = self._draw(Path(tmp), show_axons=False)
        self.assertEqual(len(collections), 2, "下敷きが消えていない")


if __name__ == "__main__":
    unittest.main()
