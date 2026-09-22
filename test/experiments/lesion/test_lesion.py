"""`NetworkBuilder.replace_global_coo()` — 損傷実験の構造的除去の土台。

`test_global_coo.py` と同じく `_generate_global_matrices()` までで止めるので、
GeNN のコンパイルは走らない。
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

root_path = Path(__file__).resolve().parents[3]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.core.NetworkBuilder import GlobalCOO, NetworkBuilder
from src.core.config_manager import ConfigManager
TASK_PATH = root_path / "scripts" / "lesion" / "task.yaml"
# テスト用のネットワーク設定。**実験の config を借りない。**
FIXTURE_CONFIG = root_path / "test" / "experiments" / "no_space_100.yaml"

from scripts.lesion.analysis.selectors import (
    BetweenGroupSelector, HubSelector, LesionContext, SynapseSelector, combine,
    parse_cut_spec)

import src.models.neurons.akita_escape_lif  # noqa: F401
import src.models.network.connectors  # noqa: F401
import src.models.network.delays  # noqa: F401
import src.models.network.space  # noqa: F401
import src.models.network.weights  # noqa: F401
import src.models.plasticity.custom_Akita  # noqa: F401
import src.models.synapses.custom  # noqa: F401
import src.models.synapses.standard_models  # noqa: F401


def _builder():
    """テスト用フィクスチャ (N=100, area: no_space) を生成だけ通した builder。"""
    config = ConfigManager().resolve(str(FIXTURE_CONFIG), "lesion", task_path=TASK_PATH)
    builder = NetworkBuilder(config, model_name="lesion_test", code_gen_dir=None)
    builder._generate_global_matrices()
    return builder


class ReplaceGlobalCooTest(unittest.TestCase):
    def test_identity_transform_changes_nothing(self):
        """恒等変換で COO も座標も不変 = 乱数ストリームがずれていないことの pin。"""
        a, b = _builder(), _builder()
        before = a.global_coo()
        b.replace_global_coo(b.global_coo())
        after = b.global_coo()
        np.testing.assert_array_equal(before.row, after.row)
        np.testing.assert_array_equal(before.col, after.col)
        np.testing.assert_array_equal(before.weights, after.weights)
        np.testing.assert_array_equal(before.delays, after.delays)

    def test_removing_rows_keeps_row_major_order(self):
        builder = _builder()
        coo = builder.global_coo()
        keep = np.ones(coo.row.size, dtype=bool)
        keep[::5] = False
        builder.replace_global_coo(GlobalCOO(coo.row[keep], coo.col[keep],
                                             coo.weights[keep], coo.delays[keep], coo.shape))
        after = builder.global_coo()
        self.assertEqual(after.row.size, int(keep.sum()))
        keys = after.row.astype(np.int64) * coo.shape[1] + after.col.astype(np.int64)
        self.assertTrue(np.all(np.diff(keys) > 0))

    def test_preserve_fan_in_records_pre_transform_counts(self):
        """切断してもゲインが動かないよう、分母は**変換前**の本数で数える。"""
        builder = _builder()
        coo = builder.global_coo()
        before = builder._pair_synapse_counts(coo)
        keep = np.ones(coo.row.size, dtype=bool)
        keep[::3] = False
        builder.replace_global_coo(
            GlobalCOO(coo.row[keep], coo.col[keep], coo.weights[keep], coo.delays[keep],
                      coo.shape),
            preserve_fan_in_scale=True)
        self.assertEqual(builder._fan_in_reference, before)
        after = builder._pair_synapse_counts(builder.global_coo())
        self.assertTrue(any(after[k] < before[k] for k in before))

    def test_rejects_broken_coo(self):
        coo = _builder().global_coo()
        cases = {
            "ソート崩れ": GlobalCOO(coo.row[::-1], coo.col[::-1], coo.weights, coo.delays,
                                    coo.shape),
            "長さ不一致": GlobalCOO(coo.row[:5], coo.col, coo.weights, coo.delays, coo.shape),
            "0本": GlobalCOO(coo.row[:0], coo.col[:0], coo.weights[:0], coo.delays[:0],
                             coo.shape),
            "shape変更": GlobalCOO(coo.row, coo.col, coo.weights, coo.delays, (1, 1)),
            "重複ペア": GlobalCOO(np.r_[coo.row[:2], coo.row[1:2]],
                                  np.r_[coo.col[:2], coo.col[1:2]],
                                  coo.weights[:3], coo.delays[:3], coo.shape),
        }
        for label, broken in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    _builder().replace_global_coo(broken)

    def test_rejects_non_finite_weights(self):
        builder = _builder()
        coo = builder.global_coo()
        weights = coo.weights.copy()
        weights[0] = np.nan
        with self.assertRaises(ValueError):
            builder.replace_global_coo(coo._replace(weights=weights))

    def test_preserve_fan_in_alone_is_rejected(self):
        """変換しないのに fan-in を固定しても意味がないので黙って通さない。"""
        config = ConfigManager().resolve(str(FIXTURE_CONFIG), "lesion", task_path=TASK_PATH)
        builder = NetworkBuilder(config, model_name="x", code_gen_dir=None)
        with self.assertRaises(ValueError):
            builder.build(preserve_fan_in_scale=True)


class SelectorTest(unittest.TestCase):
    def setUp(self):
        self.builder = _builder()
        self.coo = self.builder.global_coo()
        self.ctx = LesionContext(coo=self.coo, layout=self.builder.layout)

    def test_hub_out_direction_cuts_only_outgoing(self):
        selection = HubSelector(metric="out_degree", top=2, direction="out").select(self.ctx)
        chosen = np.array(selection.detail["neuron_ids"])
        self.assertTrue(np.all(np.isin(self.coo.row[selection.cut], chosen)))
        self.assertTrue(selection.cut.sum() > 0)

    def test_between_group_cuts_only_cross_group(self):
        selection = BetweenGroupSelector(axis="polarity").select(self.ctx)
        labels = np.asarray(self.builder.layout.labels("polarity"))
        cross = labels[self.coo.row] != labels[self.coo.col]
        np.testing.assert_array_equal(selection.cut, cross)

    def test_explicit_synapse_selection(self):
        pairs = [(int(self.coo.row[3]), int(self.coo.col[3])),
                 (int(self.coo.row[10]), int(self.coo.col[10]))]
        selection = SynapseSelector(pairs).select(self.ctx)
        self.assertEqual(int(selection.cut.sum()), 2)
        self.assertTrue(selection.cut[3] and selection.cut[10])

    def test_missing_synapse_is_an_error_not_a_silent_noop(self):
        """タイポで「何も切らなかった」が静かに通るのが最悪の失敗。"""
        with self.assertRaises(ValueError):
            SynapseSelector([(0, 0)]).select(self.ctx)

    def test_combine_modes(self):
        a = HubSelector(metric="out_degree", top=3, direction="out").select(self.ctx)
        b = BetweenGroupSelector(axis="polarity").select(self.ctx)
        self.assertEqual(int(combine([a, b], "or").cut.sum()),
                         int((a.cut | b.cut).sum()))
        self.assertEqual(int(combine([a, b], "and").cut.sum()),
                         int((a.cut & b.cut).sum()))

    def test_spec_parsing_rejects_typos(self):
        for bad in ("hub:topk=3", "unknown:x=1", "bridge:kind=zzz",
                    "hub:metric=nonsense", "hub:direction=sideways"):
            with self.subTest(bad):
                with self.assertRaises(ValueError):
                    parse_cut_spec(bad)

    def test_spec_parsing_accepts_each_selector(self):
        self.assertEqual(parse_cut_spec("hub:metric=out_degree,top=2").metric, "out_degree")
        self.assertEqual(parse_cut_spec("bridge:kind=inter_cluster").kind, "inter_cluster")
        self.assertEqual(parse_cut_spec("between:axis=polarity").axis, "polarity")
        self.assertEqual(parse_cut_spec("synapses:pairs=1-2+3-4").pairs, [(1, 2), (3, 4)])

    def test_bridge_selector_needs_geometry(self):
        with self.assertRaises(ValueError):
            parse_cut_spec("bridge:kind=any").select(self.ctx)


class CutProfileTest(unittest.TestCase):
    """切断したものの素性が、残存群と対比した形で記録されること。"""

    def setUp(self):
        self.builder = _builder()
        self.layout = self.builder.layout
        self.total = self.builder.total_neurons
        # akita_soc.yaml は weight: constant_zero (初期重みが全 0) なので、重みの
        # 比較を意味のあるものにするため合成の重みを入れる。
        coo = self.builder.global_coo()
        rng = np.random.default_rng(0)
        self.coo = coo._replace(weights=rng.uniform(0.1, 1.0, coo.row.size))

    def _profile(self, cut, **kwargs):
        from scripts.lesion.analysis.metrics import cut_profile
        return cut_profile(self.coo, cut, self.layout, self.total,
                           include_betweenness=False, **kwargs)

    def test_cut_group_is_compared_against_the_kept_group(self):
        """本数だけでは何を失ったか決まらない。必ず残存群と並べる。"""
        cut = np.zeros(self.coo.row.size, dtype=bool)
        cut[::4] = True
        _per_synapse, comparison, summary = self._profile(cut)
        by_attribute = {row["attribute"]: row for row in comparison}
        self.assertIn("weight", by_attribute)
        self.assertIn("pre_participation", by_attribute)
        for row in comparison:
            self.assertEqual(row["cut_n"] + row["kept_n"], self.coo.row.size)
        self.assertEqual(summary["num_cut"], int(cut.sum()))
        self.assertEqual(summary["num_kept"], int((~cut).sum()))

    def test_heavy_synapses_show_up_as_a_higher_cut_mean(self):
        """重い結合を狙って切れば、比が 1 を明確に超えること。"""
        weights = self.coo.weights
        cut = weights >= np.median(weights)
        _per_synapse, comparison, _summary = self._profile(cut)
        weight_row = next(r for r in comparison if r["attribute"] == "weight")
        self.assertGreater(weight_row["ratio_cut_over_kept"], 1.0)
        self.assertGreater(weight_row["cut_mean"], weight_row["kept_mean"])

    def test_breakdown_reports_the_fraction_of_each_group_cut(self):
        """「本数」だけだと元から多い群が上位に来る。群内の割合まで出す。"""
        cut = np.zeros(self.coo.row.size, dtype=bool)
        cut[::3] = True
        _per_synapse, _comparison, summary = self._profile(cut)
        blocks = summary["by_ei_block"]
        self.assertTrue(blocks)
        for stats in blocks.values():
            self.assertLessEqual(stats["cut"], stats["total"])
            self.assertAlmostEqual(stats["fraction_of_group_cut"],
                                   stats["cut"] / stats["total"])
        self.assertAlmostEqual(sum(s["cut"] for s in blocks.values()), int(cut.sum()))

    def test_per_synapse_arrays_cover_only_the_cut_ones(self):
        cut = np.zeros(self.coo.row.size, dtype=bool)
        cut[:7] = True
        per_synapse, _comparison, _summary = self._profile(cut)
        for name, values in per_synapse.items():
            self.assertEqual(len(values), 7, msg=name)
        np.testing.assert_array_equal(per_synapse["row"], self.coo.row[cut])
        np.testing.assert_array_equal(per_synapse["weight"], self.coo.weights[cut])
        self.assertIn("pre_role", per_synapse)

    def test_weight_lost_fraction_is_a_share_of_total_weight(self):
        cut = np.ones(self.coo.row.size, dtype=bool)
        _per_synapse, _comparison, summary = self._profile(cut)
        self.assertAlmostEqual(summary["weight_lost_fraction"], 1.0)
        self.assertAlmostEqual(summary["fraction_cut"], 1.0)

    def test_formatted_table_names_both_groups(self):
        from scripts.lesion.analysis.metrics import format_cut_profile
        cut = np.zeros(self.coo.row.size, dtype=bool)
        cut[::5] = True
        _per_synapse, comparison, summary = self._profile(cut)
        text = format_cut_profile(comparison, summary)
        self.assertIn("切断群", text)
        self.assertIn("残存群", text)
        self.assertIn("weight", text)


class LesionFigureDataTest(unittest.TestCase):
    """図の元データの集め方。損傷実験に固有の事情を pin する。"""

    def setUp(self):
        self.layout = _builder().layout
        self.total = self.layout.total_neurons

    def _write_run(self, run_dir: Path, windows):
        """spikes_p*.npz だけの最小の run を作る。

        **窓の素性は npz が持つ。** 以前は `probes.csv` を別に置いて index で引いて
        いたが、窓幅も phase も原点も同じファイルに入れれば読み手は 1 つ開くだけで済む。
        """
        from scripts.lesion.store import records
        for index, window_ms in enumerate(windows):
            # 全ニューロンが 1 発ずつ撃つ = レートは 1000/window_ms [Hz] になるはず
            records.save_spikes(
                run_dir / records.probe_filename(records.SPIKES, index),
                np.zeros(self.total), np.arange(self.total),
                record_start_ms=(-window_ms if index == 0
                                 else float(index - 1) * records.MS_PER_HOUR),
                record_window_ms=window_ms,
                phase=records.PHASE_BASELINE if index == 0 else records.PHASE_POST)

    def _windows(self, run_dir: Path):
        """`_write_run` が書いたものを、時刻順に並べた (hour, window_ms) で読み返す。"""
        from scripts.lesion.store import records
        found = [(records.read_window_meta(probe.path), probe.index)
                 for probe in records.discover_probes(run_dir, records.SPIKES)]
        found.sort(key=lambda item: item[0][0])
        return [(start / records.MS_PER_HOUR, window_ms, phase)
                for (start, window_ms, phase), _index in found]

    def test_firing_rate_uses_each_probes_own_window(self):
        """baseline と post で窓幅が違っても、窓の違いが発火率の段差にならないこと。

        config の `record_window_ms` 1 つで割る develop の fig2d のやり方は損傷実験では
        使えない (baseline の窓を別に取れるため)。窓幅は **その窓の npz** が持つ。
        """
        from src.utils.analysis.spikes import firing_rates
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            self._write_run(run_dir, [2000.0, 1000.0, 1000.0])
            windows = self._windows(run_dir)

            rates = [firing_rates(np.arange(self.total), self.total, window_ms)
                     for _hour, window_ms, _phase in windows]
            # 1 発 / 2 秒 = 0.5 Hz、1 発 / 1 秒 = 1.0 Hz
            np.testing.assert_allclose(rates[0], 0.5)
            np.testing.assert_allclose(rates[1], 1.0)

    def test_probes_are_ordered_by_time_not_by_filename(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            self._write_run(run_dir, [1000.0, 1000.0, 1000.0])
            hours = [hour for hour, _w, _p in self._windows(run_dir)]
            self.assertTrue(np.all(np.diff(hours) > 0))
            self.assertLess(hours[0], 0.0)   # baseline は切断より前

    def test_missing_probes_are_reported_not_guessed(self):
        """probe が 1 つも無い run は `open_run()` が落とす。**推測して続行しない。**"""
        from scripts.lesion.store import records
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(records.discover_probes(Path(tmp_dir), records.SPIKES), [])


class ProbeIoTest(unittest.TestCase):
    def test_pre_lesion_weights_are_not_mistaken_for_a_probe(self):
        """probe 以外の記録が probe の glob に拾われないこと (回帰テスト)。

        切断前の結合は以前 `weights_pre.npz` という名前で、`weights_p*.npz` の glob に
        引っかかって `discover_probes` が落ちていた。いまは `connectivity_pre.npz`。
        """
        from scripts.lesion.store.records import (PRE_CONNECTIVITY_NAME, WEIGHTS,
                                                  discover_probes, probe_filename)
        self.assertFalse(PRE_CONNECTIVITY_NAME.startswith("weights_p"))
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            np.savez_compressed(run_dir / PRE_CONNECTIVITY_NAME, data=np.array([1.0]))
            for index in (0, 1):
                np.savez_compressed(run_dir / probe_filename(WEIGHTS, index),
                                    data=np.array([1.0]))
            self.assertEqual([p.index for p in discover_probes(run_dir, WEIGHTS)], [0, 1])

    def test_probe_names_do_not_collide_with_akita_records(self):
        """akita_soc の解析 CLI が損傷 run を誤読しないこと。"""
        from scripts.develop.store.records import RECORD_PATTERN, record_glob
        from scripts.lesion.store.records import (discover_probes, parse_probe_name,
                                                  probe_filename)
        name = probe_filename("spikes", 3)
        self.assertEqual(name, "spikes_p003.npz")
        self.assertIsNone(RECORD_PATTERN.fullmatch(name))
        self.assertEqual(parse_probe_name(name), ("spikes", 3))
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            for index in (2, 0, 10):
                np.savez_compressed(run_dir / probe_filename("spikes", index), x=np.array([1]))
            self.assertEqual([p.index for p in discover_probes(run_dir, "spikes")], [0, 2, 10])
            self.assertEqual(list(run_dir.glob(record_glob("spikes"))), [])


if __name__ == "__main__":
    unittest.main()
