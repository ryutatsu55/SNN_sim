"""develop 実験 (`scripts/develop/`) の固定したい振る舞い。

- 論文の「100」は定数ではなく系のサイズ N であること
- module 軸を持たない run でも記録時刻に到達したときに落ちないこと
- seed の範囲指定が意図どおり展開されること
- `config.yaml` が「build を通った run の記録」以外にならないこと
- 指標の列が 1 通りしかないこと (本番と再解析で食い違わない)
- 再解析が、figure が読む場所に figure が読む名前で指標を置くこと
- 記録時刻の正が npz にあり、ファイル名の丸めに影響されないこと
"""
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

root_path = Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from scripts.develop.analysis import metrics
from scripts.develop.figures import style
from scripts.develop.store import paths
from scripts.develop.store.records import METRICS_NAME, MS_PER_HOUR, SPIKES, WEIGHTS, \
    record_filename, save_spikes, save_weight_values
from scripts.develop.replot import replot
from scripts.develop.store.series import open_run
from src.core.config_manager import (ConfigManager, expand_hours_spec,
                                     expand_seed_spec, load_yaml)
from src.core.layout import NetworkLayout
from src.core.config_manager import CONFIG_NAME
from scripts.develop.store.paths import DATA_SUBDIR
from scripts.develop.store.records import CONNECTIVITY_NAME

TASK_PATH = root_path / "scripts" / "develop" / "task.yaml"
# テスト用のネットワーク設定。**実験の config を借りない** ——
# 借りると、実験の条件を変えるたびにテストが落ちる。
FIXTURE_CONFIG = root_path / "test" / "experiments" / "no_space_100.yaml"


def _resolved_config(path=FIXTURE_CONFIG):
    return ConfigManager().resolve(str(path), "develop", task_path=TASK_PATH)


def _fake_coo(num_neurons: int = 100, fan_out: int = 3, seed: int = 1):
    """試験用の COO。**本物の run と同じく行優先ソート済みで、(pre, post) の重複なし。**

    重複を作らないために行ごとに非復元抽出する。ここが崩れていると、記録を読む側の
    行優先チェック (`records._require_row_major`) が正しく弾いてしまい、テストが
    「規約を守っていない入力」を使っていることになる。
    """
    rng = np.random.default_rng(seed)
    row = np.repeat(np.arange(num_neurons), fan_out)
    col = np.concatenate([np.sort(rng.choice(num_neurons, size=fan_out, replace=False))
                          for _ in range(num_neurons)])
    weights = rng.uniform(0.0, 1.0, row.size)
    return row, col, weights


def _make_run(tmp_dir, hours=(0.0,)) -> Path:
    """再解析にかけられる最小の run を作る (GeNN は通さない)。

    新しいレイアウトで置く: `config.yaml` は run 直下、npz は `data/`。
    """
    run_dir = Path(tmp_dir)
    paths.prepare(run_dir)

    manager = ConfigManager()
    config = manager.resolve(str(FIXTURE_CONFIG), "develop",
                             task_path=TASK_PATH)
    config.task.record_window_ms = 30000.0
    # 本来 NetworkBuilder が build() 時に焼き込む値。ここはフィクスチャで
    # ビルドを通さないので、save_config() の警告を出さないために実値を入れておく。
    config.network.sparse = "off"
    manager.save_config(config, save_dir=run_dir)

    row, col, weights = _fake_coo()
    np.savez_compressed(paths.data_path(run_dir, CONNECTIVITY_NAME),
                        row=row, col=col, shape=(100, 100))

    rng = np.random.default_rng(0)
    for hour in hours:
        start_ms = hour * MS_PER_HOUR
        save_spikes(
            paths.data_path(run_dir, record_filename(SPIKES, hour)),
            times=start_ms + np.sort(rng.uniform(0.0, 30000.0, 500)),
            ids=rng.integers(0, 100, 500),
            record_start_ms=start_ms,
        )
        save_weight_values(paths.data_path(run_dir, record_filename(WEIGHTS, hour)),
                           weights)
    return run_dir


class AvalancheSmaxTest(unittest.TestCase):
    """論文の [1, 100] は定数ではなく系のサイズ N。"""

    def test_defaults_to_network_size(self):
        config = _resolved_config()
        self.assertEqual(config.simulation.N, 100)
        # 論文条件 (N=100) では従来と同一の値になる = 既存結果と比較可能
        self.assertEqual(metrics.resolve_avalanche_smax(config), 100)

    def test_follows_n_for_other_configs(self):
        config = _resolved_config("configs/axon_growth_grid2.yaml")
        self.assertEqual(metrics.resolve_avalanche_smax(config), config.simulation.N)
        self.assertNotEqual(metrics.resolve_avalanche_smax(config), 100)

    def test_override_wins(self):
        config = _resolved_config()
        self.assertEqual(metrics.resolve_avalanche_smax(config, 250), 250)

    def test_rejects_degenerate_override(self):
        config = _resolved_config()
        with self.assertRaises(ValueError):
            metrics.resolve_avalanche_smax(config, 1)


class OrderAxesFallbackTest(unittest.TestCase):
    """module 軸を持たない run で描画が KeyError を投げないこと。

    記録ループはシミュレーションの途中で図を描くので、ここで例外が出ると
    **その時点までの数時間の実行が失われる**。
    """

    def test_drops_axis_the_layout_does_not_have(self):
        # フィクスチャは area: no_space なので module 軸を持たない
        layout = NetworkLayout.from_config(_resolved_config())
        self.assertFalse(layout.has_axis("module"))
        self.assertEqual(style.available_order_axes(layout), style.FALLBACK_ORDER_AXES)

    def test_keeps_axes_that_exist(self):
        layout = NetworkLayout.from_config(_resolved_config())
        layout.add_axis("module", np.array(["M0"] * layout.total_neurons))
        self.assertEqual(style.available_order_axes(layout), style.ORDER_AXES)

    def test_no_layout_means_no_ordering(self):
        self.assertIsNone(style.available_order_axes(None))


class TaskSelectionTest(unittest.TestCase):
    """どの記録プロトコルで走るかはメイン config の `task:` が決めること。

    記録条件は結果を変えるので、選択が呼び出し側 (CLI や スクリプトのハードコード) に
    あると、`config.yaml` を見ても何で走ったか分からなくなる。
    """

    def test_task_comes_from_the_main_config(self):
        """解決されたプロファイルが、メイン config の `task:` が名指したものであること。

        **名前を決め打ちしない。** どのプロファイルで走らせるかは実験条件で動かす値で、
        ここが守るのは「選んでいるのが config であって呼び出し側ではない」こと。
        """
        main_config = root_path / "scripts" / "develop" / "axon_growth_hierarchy.yaml"
        declared = load_yaml(main_config)["task"]
        config = ConfigManager().resolve(str(main_config), task_path=TASK_PATH)
        self.assertEqual(config.task.profile_name, declared)

    def test_config_without_task_is_rejected(self):
        # `task:` を書いていない config は、既定を推測せず落ちる
        with self.assertRaises(ValueError):
            ConfigManager().resolve(str(FIXTURE_CONFIG),
                                    task_path=TASK_PATH)

    def test_explicit_argument_still_wins(self):
        # 1 つの config を複数 task で使い回す既存スクリプトのための経路
        config = ConfigManager().resolve(
            str(root_path / "scripts" / "develop" / "axon_growth_hierarchy.yaml"),
            "develop", task_path=TASK_PATH)
        self.assertEqual(config.task.profile_name, "develop")


class TraceSettingTest(unittest.TestCase):
    """膜電位トレースを採ったかどうかが、その run の記録に残ること。

    引数で渡せるようにすると、完走した run に `trace_*.npz` が無いときに
    「採らない設定だった」のか「採ろうとして失敗した」のかが区別できなくなる。
    """

    def test_task_profile_declares_it(self):
        # キーが task.yaml に無いと、run の config.yaml にも残らない
        config = _resolved_config()
        self.assertIn("trace_neuron", config.task.model_dump())
        self.assertIn("trace_window_s", config.task.model_dump())

    def test_default_is_off(self):
        from scripts.develop.run_one import resolve_trace
        config = _resolved_config()
        neuron, window_s = resolve_trace(config)
        self.assertIsNone(neuron)
        self.assertGreater(window_s, 0.0)

    def test_reads_the_neuron_id(self):
        from scripts.develop.run_one import resolve_trace
        config = _resolved_config()
        config.task.trace_neuron = 7
        self.assertEqual(resolve_trace(config)[0], 7)

    def test_rejects_out_of_range_neuron(self):
        """長い run を回し切ってから IndexError で落ちないこと。"""
        from scripts.develop.run_one import resolve_trace
        config = _resolved_config()
        config.task.trace_neuron = config.simulation.N     # 0..N-1 なので範囲外
        with self.assertRaises(SystemExit):
            resolve_trace(config)

    def test_old_config_without_the_key_still_runs(self):
        from scripts.develop.run_one import resolve_trace
        config = _resolved_config()
        del config.task.trace_neuron        # このキーが無い時代の run
        self.assertIsNone(resolve_trace(config)[0])


class SeedSpecTest(unittest.TestCase):
    """`seed` は数値なら 1 本、`1..10` なら範囲。**書き方は record_hours と同じ。**"""

    def test_scalar(self):
        self.assertEqual(expand_seed_spec(5), [5])

    def test_plain_list_stays_literal(self):
        self.assertEqual(expand_seed_spec([1, 5, 42]), [1, 5, 42])

    def test_inclusive_range(self):
        self.assertEqual(expand_seed_spec(["1..10"]), list(range(1, 11)))

    def test_bare_range_without_a_list(self):
        self.assertEqual(expand_seed_spec("1..10"), list(range(1, 11)))

    def test_range_with_step(self):
        self.assertEqual(expand_seed_spec(["1..10:2"]), [1, 3, 5, 7, 9])

    def test_mixed_and_deduplicated(self):
        self.assertEqual(expand_seed_spec([42, "5..7", 1, 6]), [1, 5, 6, 7, 42])

    def test_returns_ints(self):
        # run ディレクトリ名 (seedNN) と config.yaml に入るので float では困る
        self.assertTrue(all(isinstance(s, int) for s in expand_seed_spec(["1..3"])))

    def test_rejects_a_non_integer_seed(self):
        with self.assertRaises(ValueError):
            expand_seed_spec([1.5])

    def test_rejects_a_malformed_range(self):
        with self.assertRaises(ValueError):
            expand_seed_spec(["1-10"])

    def test_rejects_reversed_range(self):
        with self.assertRaises(ValueError):
            expand_seed_spec(["10..1"])

    def test_rejects_non_positive_step(self):
        with self.assertRaises(ValueError):
            expand_seed_spec(["1..10:0"])


class RecordHoursSpecTest(unittest.TestCase):
    """`record_hours` は数値なら 1 点、`0..12` なら範囲。**リストは範囲にならない。**"""

    def test_plain_list_stays_literal(self):
        # seed と違ってここが範囲にならないことが、この記法の前提
        self.assertEqual(expand_hours_spec([0, 6, 72]), [0.0, 6.0, 72.0])

    def test_scalar(self):
        self.assertEqual(expand_hours_spec(12), [12.0])

    def test_inclusive_range(self):
        self.assertEqual(expand_hours_spec(["0..12"]), [float(h) for h in range(13)])

    def test_range_with_step(self):
        self.assertEqual(expand_hours_spec(["0..3:0.5"]),
                         [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    def test_bare_range_without_a_list(self):
        self.assertEqual(expand_hours_spec("0..2"), [0.0, 1.0, 2.0])

    def test_mixed_and_deduplicated(self):
        self.assertEqual(expand_hours_spec([24, "1..3", 0, 2]), [0.0, 1.0, 2.0, 3.0, 24.0])

    def test_step_does_not_accumulate_float_error(self):
        # 0.1 刻みの足し算をそのまま返すと 0.30000000000000004 が出る
        self.assertEqual(expand_hours_spec(["0..0.5:0.1"]),
                         [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    def test_non_integer_literals_are_not_rounded(self):
        # 5 s = 1/720 h。テスト config が使う値がそのまま通ること
        self.assertEqual(expand_hours_spec([0, 0.0013888888888888889]),
                         [0.0, 0.0013888888888888889])

    def test_rejects_a_malformed_range(self):
        with self.assertRaises(ValueError):
            expand_hours_spec(["0-12"])

    def test_rejects_reversed_range(self):
        with self.assertRaises(ValueError):
            expand_hours_spec(["12..0"])

    def test_rejects_non_positive_step(self):
        with self.assertRaises(ValueError):
            expand_hours_spec(["0..12:0"])


class TaskProfileTest(unittest.TestCase):
    """範囲記法を展開するのは resolve()。読む側は展開後の時刻しか見ない。"""

    def _resolve_with(self, record_hours: str):
        # **実験の task.yaml を借りない** —— 借りると記録時刻を変えるたびにここが落ちる
        with tempfile.TemporaryDirectory() as tmp:
            task_path = Path(tmp) / "task.yaml"
            task_path.write_text(
                "develop:\n"
                "  duration: 43800000.0\n"
                f"  record_hours: {record_hours}\n"
                "  record_window_ms: 600000.0\n"
                "  record_buffer_ms: 10000.0\n"
                "  trace_neuron: null\n"
                "  trace_window_s: 10.0\n",
                encoding="utf-8",
            )
            config = ConfigManager().resolve(str(FIXTURE_CONFIG), "develop",
                                             task_path=task_path)
            return list(config.task.record_hours)

    def test_resolve_expands_the_range(self):
        self.assertEqual(self._resolve_with("[0..12]"), [float(h) for h in range(13)])

    def test_resolve_leaves_a_plain_list_alone(self):
        self.assertEqual(self._resolve_with("[0, 6, 72]"), [0.0, 6.0, 72.0])


class RunPathsTest(unittest.TestCase):
    """run の内部構造を知るのが paths.py だけであること。"""

    def test_records_live_under_data(self):
        self.assertEqual(paths.data_path("run", "spikes_0h.npz"),
                         Path("run") / DATA_SUBDIR / "spikes_0h.npz")

    def test_rejects_unknown_figure_kind(self):
        with self.assertRaises(ValueError):
            paths.fig_path("run", "histogram", "x.png")


class PendingConfigTest(unittest.TestCase):
    """`config.yaml` は「build を通った run の記録」以外にならないこと。

    ランチャは起動前に config を置くが、その時点では `network.sparse` がまだ "auto" で、
    記録としては不完全。だから引き継ぎは別名 (`pending_config.yaml`) で置き、
    `config.yaml` を書くのは build を通した run 本体だけにしてある。
    """

    def test_handoff_is_not_named_config_yaml(self):
        self.assertNotEqual(paths.PENDING_CONFIG_NAME, CONFIG_NAME)

    def test_dump_config_does_not_claim_to_be_a_record(self):
        """`dump_config` は sparse 未解決でも警告を出さない (記録を主張しないので)。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = _resolved_config()
            self.assertNotIn(config.network.sparse, ("on", "off"))   # まだ "auto"
            with warnings.catch_warnings():
                warnings.simplefilter("error")   # 警告が出たら失敗
                ConfigManager.dump_config(config, Path(tmp) / paths.PENDING_CONFIG_NAME)

    def test_save_config_still_warns_before_build(self):
        """`save_config` の不変条件は残っていること (引数で黙らせられない)。"""
        with tempfile.TemporaryDirectory() as tmp:
            config = _resolved_config()
            with self.assertWarns(UserWarning):
                ConfigManager().save_config(config, save_dir=tmp)

    def test_save_config_is_quiet_once_sparse_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _resolved_config()
            config.network.sparse = "off"        # build() が焼き込む値
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ConfigManager().save_config(config, save_dir=tmp)


class MetricsColumnsTest(unittest.TestCase):
    """`build_row` が **1 通りの列**を作ること。

    指標の実装が 2 つあった頃は、再解析側にだけ E/I 列と重みブロック列が無く、
    本番と再解析で列が食い違っていた。いまは入力がすべて必須なので、作られる列は
    常に同じ 1 組になる。
    """

    def _row(self, tmp_dir):
        return metrics.build_row(open_run(_make_run(tmp_dir)).windows[0])

    def test_row_has_every_family_of_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            row = self._row(tmp_dir)
        self.assertIn("delta_cr", row)                                        # スパイク系
        self.assertTrue(any(k.startswith(("exc", "inh")) for k in row))       # E/I 系
        self.assertIn("weight_at_max_fraction", row)                          # 重みブロック系
        self.assertIn("diagnosis", row)                                       # 診断

    def test_parameters_come_from_the_config(self):
        """`smax` は引数ではなく config から導かれること。

        図と指標が別々に smax を決められる状態だと、図に書かれた α と CSV の α が
        食い違う (どちらも例外を出さない)。`build_row` は window 1 つしか受け取らない。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            window = open_run(_make_run(tmp_dir)).windows[0]
            row = metrics.build_row(window)
        self.assertEqual(row["avalanche_smax"], window.config.simulation.N)


class RunSeriesTest(unittest.TestCase):
    """run 全体を読む入口 (`series.open_run`) が守ること。"""

    def test_hour_comes_from_the_npz_not_the_filename(self):
        """記録時刻の正は npz。ファイル名の `{hour:g}` は有効数字 6 桁しか持たない。

        ここが崩れると、非整数の record_hours で本番と再解析の窓の原点が約 1 ms ずれ、
        burstiness_index のビン割りが変わって値が食い違う。
        """
        start_ms = MS_PER_HOUR / 3.0            # 1/3 h = 1200000.0 ms
        exact_hour = start_ms / MS_PER_HOUR
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = _make_run(tmp_dir, hours=(exact_hour,))
            series = open_run(run_dir)
            window, = series.windows
            # ファイル名は丸められている
            self.assertEqual(window.spikes_path.name, "spikes_0.333333h.npz")
            self.assertNotEqual(float("0.333333"), exact_hour)
            # 読み戻した時刻は丸められていない
            self.assertEqual(window.hour, exact_hour)
            self.assertEqual(window.record_start_ms, start_ms)

    def test_windows_are_sorted_by_time(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = _make_run(tmp_dir, hours=(0.0, 2.0, 10.0))
            series = open_run(run_dir)
            self.assertEqual([w.hour for w in series.windows], [0.0, 2.0, 10.0])

    def test_times_are_local_to_the_window(self):
        """契約どおり `Spikes.times` が窓の先頭を 0 とするローカル時刻であること。

        絶対時刻が混ざると、アバランチ分割は同じでも burstiness のビン割りがずれる。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            window, = open_run(_make_run(tmp_dir, hours=(2.0,))).windows
            spikes = window.spikes()
            self.assertTrue(np.all(spikes.times >= 0.0))
            self.assertTrue(np.all(spikes.times <= window.record_window_ms))
            self.assertEqual(window.record_start_ms, 2.0 * MS_PER_HOUR)

    def test_missing_config_is_loud(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = _make_run(tmp_dir)
            (run_dir / CONFIG_NAME).unlink()
            with self.assertRaises(FileNotFoundError):
                open_run(run_dir)

    def test_unsorted_connectivity_is_rejected(self):
        """COO の並びを揃える前に作られた run を、読んだ時点で弾くこと。

        本数も値も正しいので位置で対応づけると**黙って別のシナプスに重みが乗る**。
        静かな誤りにせず、読み込みで止める。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = _make_run(tmp_dir)
            row, col, _ = _fake_coo()
            shuffled = np.random.default_rng(0).permutation(row.size)
            np.savez_compressed(paths.data_path(run_dir, CONNECTIVITY_NAME),
                                row=row[shuffled], col=col[shuffled], shape=(100, 100))
            with self.assertRaises(ValueError) as caught:
                open_run(run_dir)
            self.assertIn("行優先", str(caught.exception))

    def test_missing_connectivity_is_loud(self):
        """古い密形式の run は、列を減らして続行せず落ちること。"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = _make_run(tmp_dir)
            paths.data_path(run_dir, CONNECTIVITY_NAME).unlink()
            with self.assertRaises(FileNotFoundError):
                open_run(run_dir)

    def test_weight_count_mismatch_is_loud(self):
        """重みの本数が connectivity と合わなければ落ちること。

        黙って通すと、ブロック分けが 1 本ずつずれた図と指標ができあがる。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = _make_run(tmp_dir)
            series = open_run(run_dir)
            save_weight_values(series.windows[0].weights_path, np.zeros(3))
            with self.assertRaises(ValueError):
                series.windows[0].weights()


class ReplotPlacementTest(unittest.TestCase):
    """再解析した指標が figure の読む場所に、figure が読む名前で置かれること。

    ここがズレると再解析は指標を計算し直すのに図は古い `metrics.csv` のまま、という
    **例外の出ない**食い違いになる。本番と同じ `metrics.csv` を上書きするので、
    置き場所を間違えると古い CSV がそのまま残ることになる。
    """

    def _assert_placement(self, run_dir: Path):
        metrics_path = paths.data_path(run_dir, METRICS_NAME)
        self.assertTrue(metrics_path.exists(), f"{metrics_path} が無い")
        header = metrics_path.read_text(encoding="utf-8").splitlines()[0]
        # fig2c の天井線がこの列を見る
        self.assertIn("avalanche_smax", header)
        # 重みブロック列まで揃っていること (本番と同じ列)
        self.assertIn("weight_at_max_fraction", header)
        # 図は種類別のサブディレクトリへ
        self.assertTrue(paths.fig_path(run_dir, paths.RASTER, "raster_0h.png").exists())
        self.assertTrue(paths.fig_path(run_dir, paths.AVALANCHE, "avalanche_0h.png").exists())

    def test_writes_where_the_figures_read(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = _make_run(tmp_dir)
            replot(run_dir, with_structure=False)
            self._assert_placement(run_dir)

    def test_rebuild_must_match_the_recorded_connectivity(self):
        """構造図のための再ビルドが、記録と違うネットワークを作ったら落ちること。

        このフィクスチャの COO は本物のビルド結果ではないので、再ビルドすれば必ず
        食い違う。**黙って別のネットワークの構造図を描かない**ことをここで押さえる
        (他のテストが `with_structure=False` を渡しているのはこのため)。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = _make_run(tmp_dir)
            with self.assertRaises(SystemExit):
                replot(run_dir)

    def test_metrics_hour_matches_the_series(self):
        """`metrics.csv` の hour 列と npz 由来の時刻が一致すること。

        食い違うと fig2c が「指標は N 行、重み軌跡は M 点」の図を描こうとする。
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = _make_run(tmp_dir, hours=(0.0, 6.0))
            replot(run_dir, with_structure=False)
            rows = paths.data_path(run_dir, METRICS_NAME).read_text(
                encoding="utf-8").splitlines()
            column = rows[0].split(",").index("hour")
            recorded = [float(row.split(",")[column]) for row in rows[1:]]
            self.assertEqual(recorded, [w.hour for w in open_run(run_dir).windows])


if __name__ == "__main__":
    unittest.main()
