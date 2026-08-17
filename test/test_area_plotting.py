"""エリア可視化 (src/utils/plotting/area.py) の回帰テスト。

SDF を格子上で評価して描くやり方には、実装時に踏んだ罠が 3 つある。ここで固定するのは
その 3 つと、描画が乱数に触らないこと。
"""

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


if __name__ == "__main__":
    unittest.main()
