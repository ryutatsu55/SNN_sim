"""`scripts/develop.py` の N 非依存化まわり。

このスクリプトは akita_soc_fig2.py のフォークだが、N=100 以外の config も回す前提なので
「論文の 100」が N に追従すること、および module 軸を持たない run で長い実行が
落ちないことを固定する。
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from scripts.develop import (
    FALLBACK_ORDER_AXES,
    RASTER_ORDER_AXES,
    REPLOT_METRICS_NAME,
    raster_ylim,
    replot_existing_output,
    resolve_avalanche_smax,
    resolve_order_axes,
)
from src.core.config_manager import ConfigManager
from src.core.layout import NetworkLayout
from src.core.output_manager import CONFIG_NAME, DATA_SUBDIR, data_dir


def _resolved_config(path="configs/akita_soc.yaml"):
    return ConfigManager().resolve(str(root_path / path), "akita_soc_fig2")


class AvalancheSmaxTest(unittest.TestCase):
    """論文の [1, 100] は定数ではなく系のサイズ N。"""

    def test_defaults_to_network_size(self):
        config = _resolved_config()
        self.assertEqual(config.simulation.N, 100)
        # 論文条件 (N=100) では従来と同一の値になる = 既存結果と比較可能
        self.assertEqual(resolve_avalanche_smax(config), 100)

    def test_follows_n_for_other_configs(self):
        config = _resolved_config("configs/axon_growth_grid2.yaml")
        self.assertEqual(resolve_avalanche_smax(config), config.simulation.N)
        self.assertNotEqual(resolve_avalanche_smax(config), 100)

    def test_cli_override_wins(self):
        config = _resolved_config()
        self.assertEqual(resolve_avalanche_smax(config, 250), 250)

    def test_rejects_degenerate_override(self):
        config = _resolved_config()
        with self.assertRaises(ValueError):
            resolve_avalanche_smax(config, 1)

    def test_raster_ylim_tracks_n(self):
        self.assertEqual(raster_ylim(100), (0.0, 100.0))   # 論文条件は不変
        self.assertEqual(raster_ylim(256), (0.0, 256.0))


class OrderAxesFallbackTest(unittest.TestCase):
    """module 軸を持たない run で `plot_raster` が KeyError を投げないこと。

    develop.py の記録ループはシミュレーションの途中で図を描くので、ここで例外が出ると
    **その時点までの数時間の実行が失われる**。
    """

    def test_drops_axis_the_layout_does_not_have(self):
        # akita_soc.yaml は area: no_space なので module 軸を持たない
        layout = NetworkLayout.from_config(_resolved_config())
        self.assertFalse(layout.has_axis("module"))
        self.assertEqual(resolve_order_axes(layout), FALLBACK_ORDER_AXES)

    def test_keeps_axes_that_exist(self):
        layout = NetworkLayout.from_config(_resolved_config())
        layout.add_axis("module", np.array(["M0"] * layout.total_neurons))
        self.assertEqual(resolve_order_axes(layout), RASTER_ORDER_AXES)

    def test_no_layout_means_no_ordering(self):
        self.assertIsNone(resolve_order_axes(None))


class ReplotOutputPlacementTest(unittest.TestCase):
    """再解析した指標が fig2c の読む場所に、fig2c が読む名前で置かれること。

    ここがズレると `--replot-from` は指標を計算し直すのに図は古い metrics.csv のまま、
    という**例外の出ない**食い違いになる。
    """

    def _make_run(self, tmp_dir, organized: bool) -> Path:
        run_dir = Path(tmp_dir)
        target = run_dir / DATA_SUBDIR if organized else run_dir
        target.mkdir(parents=True, exist_ok=True)

        manager = ConfigManager()
        config = manager.resolve(str(root_path / "configs" / "akita_soc.yaml"), "akita_soc_fig2")
        config.task.record_window_ms = 30000.0
        # 本来 NetworkBuilder が build() 時に焼き込む値。ここは replot のフィクスチャで
        # ビルドを通さないので、save_config() の警告を出さないために実値を入れておく。
        config.network.sparse = "off"
        manager.save_config(config, save_dir=target)

        rng = np.random.default_rng(0)
        np.savez_compressed(
            target / "spikes_0h.npz",
            times=np.sort(rng.uniform(0.0, 30000.0, 500)),
            ids=rng.integers(0, 100, 500),
        )
        return run_dir

    def _assert_placement(self, run_dir: Path):
        metrics_path = data_dir(run_dir) / REPLOT_METRICS_NAME
        self.assertTrue(metrics_path.exists(), f"{metrics_path} が無い")
        header = metrics_path.read_text(encoding="utf-8").splitlines()[0]
        # fig2c の天井線がこの列を見る
        self.assertIn("avalanche_smax", header)
        # 図は run ルートへ (data/ の中ではなく)
        self.assertTrue((run_dir / "raster_0h.png").exists())
        self.assertTrue((run_dir / "avalanche_0h.png").exists())

    def test_run_root_of_organized_run(self):
        """organize_output() 後の run ルートを渡しても npz を見つけられること。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = self._make_run(tmp_dir, organized=True)
            self.assertFalse((run_dir / CONFIG_NAME).exists())  # data/ にしか無い
            replot_existing_output(run_dir)
            self._assert_placement(run_dir)

    def test_data_subdir_is_normalized_to_run_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = self._make_run(tmp_dir, organized=True)
            replot_existing_output(run_dir / DATA_SUBDIR)
            self._assert_placement(run_dir)

    def test_unorganized_run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = self._make_run(tmp_dir, organized=False)
            replot_existing_output(run_dir)
            self._assert_placement(run_dir)

    def test_smax_override_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = self._make_run(tmp_dir, organized=True)
            replot_existing_output(run_dir, smax_override=250)
            rows = (data_dir(run_dir) / REPLOT_METRICS_NAME).read_text(encoding="utf-8").splitlines()
            column = rows[0].split(",").index("avalanche_smax")
            self.assertEqual(rows[1].split(",")[column], "250")


if __name__ == "__main__":
    unittest.main()
