"""`scripts/akita_soc/` の記録規約と読み出しを検証する。

**develop と同じ形の実験なので、見るところも同じ** —— 記録ファイル名の往復、窓の原点が
npz から来ること、指標の列が揃うこと、そして `develop` の規約と衝突しないこと。

モデルと数式そのもの (escape LIF / STDP カーネル / べき乗フィット) は
`test/test_akita_soc.py` が見る。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

root_path = Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from scripts.akita_soc.analysis import metrics
from scripts.akita_soc.figures import style
from scripts.akita_soc.store import paths
from scripts.akita_soc.store.records import (MS_PER_HOUR, SPIKES, WEIGHTS, discover_records,
                                             parse_hour, record_filename, save_spikes,
                                             save_weight_values)
from scripts.akita_soc.store.series import open_run
from src.core.config_manager import ConfigManager
from scripts.akita_soc.store.records import CONNECTIVITY_NAME

TASK_PATH = root_path / "scripts" / "akita_soc" / "task.yaml"
CONFIG_PATH = root_path / "scripts" / "akita_soc" / "akita_soc.yaml"


def _fake_coo(num_neurons: int = 100, fan_out: int = 3, seed: int = 1):
    """試験用の COO。**本物の run と同じく行優先ソート済みで、(pre, post) の重複なし。**"""
    rng = np.random.default_rng(seed)
    row = np.repeat(np.arange(num_neurons), fan_out)
    col = np.concatenate([np.sort(rng.choice(num_neurons, size=fan_out, replace=False))
                          for _ in range(num_neurons)])
    weights = rng.uniform(0.0, 1.0, row.size)
    return row, col, weights


def _make_run(tmp_dir, hours=(0.0, 6.0)) -> Path:
    """読み出しにかけられる最小の run を作る (GeNN は通さない)。"""
    run_dir = Path(tmp_dir)
    paths.prepare(run_dir)

    manager = ConfigManager()
    config = manager.resolve(str(CONFIG_PATH), task_path=TASK_PATH)
    config.task.record_window_ms = 30000.0
    # 本来 NetworkBuilder が build() 時に焼き込む値。ビルドを通さないので実値を入れておく。
    config.network.sparse = "off"
    manager.save_config(config, save_dir=run_dir)

    row, col, weights = _fake_coo()
    np.savez_compressed(paths.data_path(run_dir, CONNECTIVITY_NAME),
                        row=row, col=col, shape=(100, 100))

    rng = np.random.default_rng(0)
    for hour in hours:
        start_ms = hour * MS_PER_HOUR
        save_spikes(paths.data_path(run_dir, record_filename(SPIKES, hour)),
                    times=start_ms + np.sort(rng.uniform(0.0, 30000.0, 500)),
                    ids=rng.integers(0, 100, 500),
                    record_start_ms=start_ms)
        save_weight_values(paths.data_path(run_dir, record_filename(WEIGHTS, hour)), weights)
    return run_dir


class TaskProfileTest(unittest.TestCase):
    """記録プロトコルはこの実験の持ち物で、メイン config の `task:` が選ぶ。"""

    def test_config_selects_its_own_profile(self):
        config = ConfigManager().resolve(str(CONFIG_PATH), task_path=TASK_PATH)
        self.assertEqual(config.task.profile_name, "akita_soc")
        self.assertEqual(list(config.task.record_hours), [0, 6, 72])

    def test_sequential_probe_profile_records_more_points(self):
        """遷移を細かく追うプロファイルが、既定より多くの点を記録すること。

        **点数そのものは決め打ちしない。** 何点採るかは実験条件で動かす値で、
        ここが守るのは「別プロファイルとして分けてある」ことの方。
        """
        default = ConfigManager().resolve(str(CONFIG_PATH), task_path=TASK_PATH)
        probe = ConfigManager().resolve(str(CONFIG_PATH), "akita_soc_sequential_probe",
                                        task_path=TASK_PATH)
        self.assertGreater(len(probe.task.record_hours), len(default.task.record_hours))


class RecordNamingTest(unittest.TestCase):
    def test_filename_round_trips(self):
        for hour in (0.0, 0.5, 6.0, 72.0, 1.25):
            name = record_filename(WEIGHTS, hour)
            self.assertEqual(parse_hour(Path(name), WEIGHTS), hour)

    def test_wrong_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_hour(Path("spikes_6h.npz"), WEIGHTS)
        for name in ("weights_6.npz", "weights.npz", "connectivity.npz", "weights_xh.npz"):
            with self.assertRaises(ValueError):
                parse_hour(Path(name))

    def test_records_are_discovered_in_time_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            for name in ("weights_72h.npz", "weights_0h.npz", "weights_6h.npz"):
                np.savez_compressed(run_dir / name, data=np.zeros(4, dtype=np.float32))
            self.assertEqual([item.hour for item in discover_records(run_dir, WEIGHTS)],
                             [0.0, 6.0, 72.0])

    def test_lesion_probes_are_not_mistaken_for_records(self):
        """損傷 run を akita_soc の replot に渡しても「見つからない」で止まること。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            np.savez_compressed(run_dir / "spikes_p003.npz", x=np.array([1]))
            self.assertEqual(discover_records(run_dir, SPIKES), [])


class OrderAxesTest(unittest.TestCase):
    """この実験は `area: no_space` なので、並べ替えは E/I だけ。"""

    def test_order_axes_is_polarity_only(self):
        self.assertEqual(style.ORDER_AXES, ("polarity",))


class SeriesTest(unittest.TestCase):
    def test_hour_comes_from_the_npz_not_the_filename(self):
        """**時刻の正は npz。** ファイル名は並べ替えと目印にしか使わない。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = _make_run(tmp_dir, hours=(0.0, 6.0))
            series = open_run(run_dir)
            self.assertEqual([window.hour for window in series.windows], [0.0, 6.0])

    def test_times_are_local_to_the_window(self):
        """契約どおり、`Spikes.times` は窓の先頭を 0 とするローカル時刻。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            series = open_run(_make_run(tmp_dir, hours=(6.0,)))
            times = series.windows[0].spikes().times
            self.assertGreaterEqual(times.min(), 0.0)
            self.assertLess(times.max(), 30000.0)


class MetricsColumnsTest(unittest.TestCase):
    def test_row_has_every_family_of_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            series = open_run(_make_run(tmp_dir, hours=(0.0,)))
            row = metrics.build_row(series.windows[0])
        for column in ("hour", "num_spikes", "mean_rate_hz", "llr", "delta_cr",
                       "burstiness_index", "bimodality_d", "avalanche_smax",
                       "weight_at_max_fraction", "diagnosis"):
            self.assertIn(column, row, f"{column} 列がありません")

    def test_smax_follows_the_network_size(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            series = open_run(_make_run(tmp_dir, hours=(0.0,)))
            self.assertEqual(metrics.resolve_avalanche_smax(series.config),
                             int(series.config.simulation.N))


if __name__ == "__main__":
    unittest.main()
