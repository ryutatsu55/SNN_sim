"""損傷実験 (`scripts/lesion/`) の切断・命名・可塑性の凍結・記録の読み書きを検証する。

- **構造的除去** (`NetworkBuilder.replace_global_coo()`): 指定したシナプスだけを除き、COO の
  不変条件 (row-major・有限の重み) を保つ。壊れた COO は弾く
- **切断 spec の選択**: 各選択子 (hub / 群間 / 明示 / random / bridge) が指定どおりのシナプスだけを選ぶ。
  存在しない対象や綴りの誤りは黙って無視せずエラーにする。random は seed で再現し、割合をまたいで入れ子になる
- **run ディレクトリ名**: 既存の run と 1 文字も変えない
- **可塑性の凍結**: 指定した群の学習率だけを 0 にし、元の config を変えない
- **駆動**: 1 発が届けるコンダクタンス × 本数を正しく積み、切ったシナプスは駆動を届けない
- **切断の素性** (`cut_profile`): 切断群を残存群と対比して記録する
- **記録の読み直し**: 切断直後の記録から読み直した `Built` が切断後の builder と同じで、構造の記録と
  比較表が記録から作り直せる。probe の窓幅・順序・番号が記録どおりで、akita_soc の記録と衝突せず、
  アバランシェのビン幅は baseline で固定される

ネットワークは `_generate_global_matrices()` までで止めるので、GeNN のコンパイルは走らない。
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

from scripts.lesion.analysis.drive import conductance_per_release, drive_columns, source_params
from scripts.lesion.protocol.cut import (
    BetweenGroupSelector, HubSelector, LesionContext, RandomSelector, SynapseSelector,
    combine, cut_label, parse_cut_spec)
from scripts.lesion.protocol.freeze import freeze_plasticity, frozen_groups, run_label

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

    def test_random_cuts_only_the_requested_polarities(self):
        is_exc = np.asarray(self.builder.layout.labels("polarity")) == "excitatory"
        selection = RandomSelector(fraction=0.3, pre="exc", post="exc").select(self.ctx)
        cut_row, cut_col = self.coo.row[selection.cut], self.coo.col[selection.cut]
        self.assertTrue(np.all(is_exc[cut_row]) and np.all(is_exc[cut_col]))
        candidates = int((is_exc[self.coo.row] & is_exc[self.coo.col]).sum())
        self.assertEqual(int(selection.cut.sum()), round(0.3 * candidates))
        self.assertEqual(selection.detail["num_candidates"], candidates)

    def test_random_pre_only_keeps_both_targets(self):
        is_exc = np.asarray(self.builder.layout.labels("polarity")) == "excitatory"
        selection = RandomSelector(fraction=0.5, pre="exc").select(self.ctx)
        self.assertTrue(np.all(is_exc[self.coo.row[selection.cut]]))
        self.assertTrue(np.any(~is_exc[self.coo.col[selection.cut]]))

    def test_random_is_reproducible_by_seed(self):
        a = RandomSelector(fraction=0.2, pre="exc", seed=3).select(self.ctx).cut
        b = RandomSelector(fraction=0.2, pre="exc", seed=3).select(self.ctx).cut
        c = RandomSelector(fraction=0.2, pre="exc", seed=4).select(self.ctx).cut
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_random_cuts_are_nested_across_fractions(self):
        # 同じ seed なら、多く切る条件は少なく切る条件の切断を必ず含む (用量反応の前提)。
        cuts = [RandomSelector(fraction=f, seed=0).select(self.ctx).cut
                for f in (0.1, 0.25, 0.5, 0.9)]
        for small, large in zip(cuts, cuts[1:]):
            self.assertTrue(np.all(large[small]))
            self.assertGreater(large.sum(), small.sum())

    def test_random_label_carries_the_whole_spec(self):
        self.assertEqual(cut_label(["random:pre=exc,post=exc,fraction=0.1,seed=2"]),
                         "random-exc-exc-f0.1-seed2")

    def test_spec_parsing_rejects_typos(self):
        for bad in ("hub:topk=3", "unknown:x=1", "bridge:kind=zzz",
                    "hub:metric=nonsense", "hub:direction=sideways",
                    "random:pre=exc", "random:fraction=0", "random:fraction=0.1,pre=E",
                    "random:fraction=0.1,frac=2"):
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


class RunLabelTest(unittest.TestCase):
    """run ディレクトリ名 (`run_label`)。**既存の run と 1 文字も変えない**ことを固定する。"""

    # 既にある run の (task.cut, task.freeze_plasticity) → ディレクトリ名。
    EXISTING = [
        (["bridge:parts=BC0-0-1"], [], "bridge-BC0-0-1"),
        (["bridge:parts=BX1-3"], [], "bridge-BX1-3"),
        (["bridge:kind=inter_cluster"], [], "bridge-inter_cluster"),
        (["random:fraction=0.025"], [], "random-any-any-f0.025-seed0"),
        (["random:pre=exc,fraction=0.1"], [], "random-exc-any-f0.1-seed0"),
        (["random:fraction=0.1"], ["from_Inh"], "random-any-any-f0.1-seed0+freeze-from_Inh"),
        (["random:fraction=0.1"], ["from_Exc"], "random-any-any-f0.1-seed0+freeze-from_Exc"),
    ]

    def test_existing_run_names_are_kept(self):
        for cut, frozen, name in self.EXISTING:
            with self.subTest(name):
                protocol = {"cut": cut, "cut_combine": "or", "freeze_plasticity": frozen}
                self.assertEqual(run_label(protocol), name)

    def test_frozen_groups_are_sorted_into_the_name(self):
        protocol = {"cut": ["random:fraction=0.1"], "cut_combine": "or",
                    "freeze_plasticity": ["from_Inh", "from_Exc"]}
        self.assertEqual(run_label(protocol),
                         "random-any-any-f0.1-seed0+freeze-from_Exc-from_Inh")

    def test_name_without_freeze_is_the_cut_label(self):
        specs = ["hub:metric=participation,top=3,direction=out", "between:axis=module"]
        self.assertEqual(run_label({"cut": specs, "cut_combine": "and"}),
                         cut_label(specs, "and"))


class FreezeTest(unittest.TestCase):
    """切断後に可塑性を止める操作 (`protocol/freeze.py`)。"""

    def _config(self):
        return ConfigManager().resolve(str(FIXTURE_CONFIG), "lesion", task_path=TASK_PATH)

    def test_only_learning_rate_of_named_group_is_zeroed(self):
        config = self._config()
        frozen = freeze_plasticity(config, ["from_Inh"])
        self.assertEqual(frozen.synapses["from_Inh"].plasticity.A_I, 0.0)
        self.assertEqual(frozen.synapses["from_Exc"].plasticity.A_E,
                         config.synapses["from_Exc"].plasticity.A_E)
        self.assertGreater(config.synapses["from_Inh"].plasticity.A_I, 0.0)   # 元は変えない
        self.assertEqual(frozen.synapses["from_Inh"].plasticity.U,
                         config.synapses["from_Inh"].plasticity.U)

    def test_unknown_group_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "config.synapses"):
            freeze_plasticity(self._config(), ["from_Nothing"])

    def test_unknown_plasticity_model_is_an_error(self):
        config = self._config()
        config.synapses["from_Inh"].plasticity.type = "static"
        with self.assertRaisesRegex(ValueError, "学習率"):
            freeze_plasticity(config, ["from_Inh"])

    def test_scalar_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "リスト"):
            frozen_groups({"freeze_plasticity": "from_Inh"})


class DriveTest(unittest.TestCase):
    """E / I 駆動が「1 発が届けるコンダクタンス × 本数」を正しく積むこと。"""

    WINDOW_MS = 1000.0

    def setUp(self):
        builder = _builder()
        self.config, self.layout = builder.config, builder.layout
        self.coo = builder.global_coo()
        self.is_exc = np.asarray(self.layout.labels("polarity")) == "excitatory"
        self.params = source_params(self.config, self.layout)
        self.per_release = conductance_per_release(self.params, float(self.config.simulation.dt))

    def _drive(self, times, ids, weights):
        return drive_columns(np.asarray(times, float), np.asarray(ids), self.coo.row,
                             self.coo.col, weights, self.layout, self.config, self.WINDOW_MS)

    def test_one_spike_delivers_u_w_gmax_times_decay_sum(self):
        pre = int(np.flatnonzero(self.is_exc)[0])
        weights = np.full(self.coo.row.size, 0.5)
        out = self._drive([10.0], [pre], weights)
        targets_e = self.coo.col[(self.coo.row == pre) & self.is_exc[self.coo.col]]
        expected = (self.params.utilization[pre] * 0.5 * self.per_release[pre]
                    * targets_e.size / self.is_exc.sum() * 1000.0 / self.WINDOW_MS)
        self.assertAlmostEqual(out["drive_exc_onto_e_ns"], expected)
        self.assertEqual(out["drive_inh_onto_e_ns"], 0.0)
        self.assertTrue(np.isnan(out["drive_ei_ratio_onto_e"]))

    def test_ratio_is_exc_over_inh(self):
        pre_e = int(np.flatnonzero(self.is_exc)[0])
        pre_i = int(np.flatnonzero(~self.is_exc)[0])
        out = self._drive([10.0, 20.0], [pre_e, pre_i], np.ones(self.coo.row.size))
        self.assertAlmostEqual(out["drive_ei_ratio_onto_e"],
                               out["drive_exc_onto_e_ns"] / out["drive_inh_onto_e_ns"])

    def test_cut_synapses_stop_delivering(self):
        pre = int(np.flatnonzero(self.is_exc)[0])
        keep = self.coo.row != pre
        out = drive_columns(np.array([10.0]), np.array([pre]), self.coo.row[keep],
                            self.coo.col[keep], np.ones(int(keep.sum())),
                            self.layout, self.config, self.WINDOW_MS)
        self.assertEqual(out["drive_exc_onto_e_ns"], 0.0)

    def test_unsupported_synapse_model_is_refused(self):
        config = self.config.model_copy(deep=True)
        next(iter(config.synapses.values())).synapse.type = "ExpCurr"
        with self.assertRaises(NotImplementedError):
            source_params(config, self.layout)


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
        return cut_profile(self.coo, cut, self.layout, self.total, **kwargs)

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
        from scripts.lesion.store import formats
        # 先頭が baseline (番号を持たない)。post は 0 始まりで record_hours の添字に一致。
        for position, window_ms in enumerate(windows):
            baseline = position == 0
            index = formats.BASELINE_INDEX if baseline else position - 1
            start_ms = -window_ms if baseline else float(index) * formats.MS_PER_HOUR
            # 全ニューロンが窓の先頭で 1 発ずつ撃つ = レートは 1000/window_ms [Hz]。
            # 時刻は切断軸なので、窓の原点を足す (save_spikes が窓の外を弾く)。
            formats.save_spikes(
                run_dir / formats.probe_filename(formats.SPIKES, index),
                np.full(self.total, start_ms), np.arange(self.total),
                record_start_ms=start_ms,
                record_window_ms=window_ms,
                phase=formats.PHASE_BASELINE if baseline else formats.PHASE_POST)

    def _windows(self, run_dir: Path):
        """`_write_run` が書いたものを、時刻順に並べた (hour, window_ms) で読み返す。"""
        from scripts.lesion.store import formats
        found = [(formats.read_window_meta(probe.path), probe.index)
                 for probe in formats.discover_probes(run_dir, formats.SPIKES)]
        found.sort(key=lambda item: item[0][0])
        return [(start / formats.MS_PER_HOUR, window_ms, phase)
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
        from scripts.lesion.store import formats
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(formats.discover_probes(Path(tmp_dir), formats.SPIKES), [])


class _StubSeries:
    """`Series.avalanche_bin_ms()` だけを借りる最小のスタブ。

    ビンの決め方は `run_dir` のファイルだけで決まる (config も layout も要らない) ので、
    run を丸ごと作らずに固定できる。
    """

    def __init__(self, run_dir: Path):
        from scripts.lesion.store.views import Series
        self.run_dir = run_dir
        self._avalanche_bin_ms = None
        self.avalanche_bin_ms = Series.avalanche_bin_ms.__get__(self)


class BuildRecordTest(unittest.TestCase):
    """切断直後の記録を書いて読み直した `Built` が、切断後の builder と同じであること。"""

    def test_built_reads_the_post_lesion_network(self):
        from scripts.lesion.store import paths
        from scripts.lesion.store.built import write_built
        from scripts.lesion.store.views import open_run

        builder = _builder()
        pre = builder.global_coo()
        keep = np.ones(pre.num_synapses, dtype=bool)
        keep[::3] = False
        weights = np.linspace(0.0, 1.0, pre.num_synapses)
        builder.replace_global_coo(GlobalCOO(pre.row[keep], pre.col[keep], weights[keep],
                                             pre.delays[keep], pre.shape))
        post = builder.global_coo()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            paths.prepare(run_dir)
            ConfigManager().save_config(builder.config, save_dir=run_dir)
            write_built(run_dir, builder, builder.layout, pre_coo=pre, keep=keep)
            series = open_run(run_dir, require_windows=False)
            coo = series.built().coo()
            np.testing.assert_array_equal(coo.row, post.row)
            np.testing.assert_array_equal(coo.col, post.col)
            np.testing.assert_array_equal(coo.weights, post.weights)
            np.testing.assert_array_equal(coo.delays, post.delays)
            np.testing.assert_array_equal(series.wiring_pre().row, pre.row)

    def _post_lesion_run(self, run_dir: Path):
        """3 本に 1 本を切った run の切断直後の記録を書き、(切断前, 切断後) の COO を返す。

        `run_one.py` と同じく `lesion_cut.npz` (切ったシナプスの素性) も書く。
        """
        from scripts.lesion.analysis.metrics import cut_profile
        from scripts.lesion.store import paths
        from scripts.lesion.store.built import write_built
        from scripts.lesion.store.formats import save_cut

        builder = _builder()
        pre = builder.global_coo()
        # 重みと遅延はシナプスごとに違う値にする (引き当てを取り違えたら表が変わるように)
        pre = pre._replace(weights=np.linspace(0.1, 1.0, pre.num_synapses),
                           delays=np.linspace(1.0, 3.0, pre.num_synapses))
        keep = np.ones(pre.num_synapses, dtype=bool)
        keep[::3] = False
        builder.replace_global_coo(GlobalCOO(pre.row[keep], pre.col[keep], pre.weights[keep],
                                             pre.delays[keep], pre.shape))
        paths.prepare(run_dir)
        ConfigManager().save_config(builder.config, save_dir=run_dir)
        write_built(run_dir, builder, builder.layout, pre_coo=pre, keep=keep)
        per_synapse, _comparison, _summary = cut_profile(
            pre, ~keep, builder.layout, builder.layout.total_neurons)
        save_cut(paths.data_dir(run_dir), per_synapse)
        return pre, builder.global_coo()

    def test_cut_profile_is_rebuilt_from_the_records(self):
        """記録から作り直した比較表が、切断の瞬間に持っていた値から作った表と同じであること。

        残存側は切断後の記録、切断側は `lesion_cut.npz` から (pre, post) で引き当てる。
        引き当てを取り違えると、例外を出さずに別の値の表になる。
        """
        from scripts.lesion.analysis.metrics import cut_profile, pre_cut_coo
        from scripts.lesion.store.views import open_run
        with tempfile.TemporaryDirectory() as tmp_dir:
            pre, _post = self._post_lesion_run(Path(tmp_dir))
            built = open_run(Path(tmp_dir), require_windows=False).built()
            coo, is_cut = pre_cut_coo(built)
            np.testing.assert_array_equal(coo.row, pre.row)
            np.testing.assert_allclose(coo.weights, pre.weights)
            np.testing.assert_allclose(coo.delays, pre.delays)
            np.testing.assert_array_equal(is_cut[::3], True)
            self.assertEqual(int(is_cut.sum()), len(range(0, pre.num_synapses, 3)))
            _p, expected, _s = cut_profile(pre, is_cut, built.layout, built.total_neurons)
            _p, rebuilt, _s = cut_profile(coo, is_cut, built.layout, built.total_neurons)
        np.testing.assert_equal(rebuilt, expected)   # nan どうしも等しいとみなす

    def test_structure_record_has_one_row_per_wiring(self):
        """`structure.csv` は切断前と切断後の 2 行で、それぞれの配線から測ってあること。

        トポロジーは切断後の run を通して変わらないので probe ごとには測らない。
        """
        from scripts.lesion.analysis.structure import structure_rows
        from scripts.lesion.store.views import open_run
        with tempfile.TemporaryDirectory() as tmp_dir:
            pre, post = self._post_lesion_run(Path(tmp_dir))
            rows = structure_rows(open_run(Path(tmp_dir), require_windows=False).built())
        self.assertEqual([row["wiring"] for row in rows], ["pre", "post"])
        self.assertEqual(rows[0]["num_synapses"], pre.num_synapses)
        self.assertEqual(rows[1]["num_synapses"], post.num_synapses)
        # 重みを使う量は probe ごとの metrics.csv の担当
        self.assertNotIn("out_strength_mean", rows[0])

    def test_structure_stage_separates_records_from_reports(self):
        """構造の段階は記録を `data/` に、人が読む表を `report/structure/` に書くこと。"""
        from scripts.lesion.report import structure
        from scripts.lesion.store import paths
        from scripts.lesion.store.formats import STRUCTURE_NAME
        from scripts.lesion.store.views import open_run
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            self._post_lesion_run(run_dir)
            structure.emit(open_run(run_dir, require_windows=False).built())
            self.assertTrue(paths.data_path(run_dir, STRUCTURE_NAME).exists())
            for name in ("connection_probability.csv", "cut_profile.csv"):
                self.assertTrue(paths.report_path(run_dir, paths.STRUCTURE, name).exists())
                self.assertFalse(paths.data_path(run_dir, name).exists())

    def test_initial_weights_are_not_mistaken_for_a_probe(self):
        """`weights_initial.npz` が probe の glob に拾われないこと。"""
        from scripts.lesion.store.formats import (INITIAL_WEIGHTS_NAME, WEIGHTS,
                                                  discover_probes)
        with tempfile.TemporaryDirectory() as tmp_dir:
            np.savez_compressed(Path(tmp_dir) / INITIAL_WEIGHTS_NAME, data=np.array([1.0]))
            self.assertEqual(discover_probes(Path(tmp_dir), WEIGHTS), [])


class ProbeIoTest(unittest.TestCase):
    def test_pre_lesion_weights_are_not_mistaken_for_a_probe(self):
        """probe 以外の記録が probe の glob に拾われないこと (回帰テスト)。

        切断前の結合は以前 `weights_pre.npz` という名前で、`weights_p*.npz` の glob に
        引っかかって `discover_probes` が落ちていた。いまは `connectivity_pre.npz`。
        """
        from scripts.lesion.store.formats import (PRE_CONNECTIVITY_NAME, WEIGHTS,
                                                  discover_probes, probe_filename)
        self.assertFalse(PRE_CONNECTIVITY_NAME.startswith("weights_p"))
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            np.savez_compressed(run_dir / PRE_CONNECTIVITY_NAME, data=np.array([1.0]))
            for index in (0, 1):
                np.savez_compressed(run_dir / probe_filename(WEIGHTS, index),
                                    data=np.array([1.0]))
            self.assertEqual([p.index for p in discover_probes(run_dir, WEIGHTS)], [0, 1])

    def test_avalanche_bin_is_fixed_to_the_baseline(self):
        """アバランチのビンが窓ごとに変わらないこと。

        Beggs-Plenz の既定は「その窓の平均イベント間隔」だが、損傷で発火率が落ちると
        ビンが伸び、それだけで σ_bp が上がって見える (実測で 0.496 -> 0.529、
        同じビンなら 0.496 -> 0.441 と符号が逆)。切断の前後を比べるのが目的なので、
        ビンは切断前で固定する。
        """
        from src.utils.analysis.avalanche import mean_iei_ms
        from scripts.lesion.store import paths, formats
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            data = paths.data_dir(run_dir)
            data.mkdir(parents=True)
            rng = np.random.default_rng(0)
            # baseline は密、post は疎 -> 窓ごとに合わせるとビンが倍以上変わる
            for index, rate in ((formats.BASELINE_INDEX, 400), (0, 100)):
                start = -1000.0 if index == formats.BASELINE_INDEX else 0.0
                times = np.sort(rng.uniform(0.0, 1000.0, rate)) + start
                formats.save_spikes(
                    data / formats.probe_filename(formats.SPIKES, index),
                    times, rng.integers(0, 10, rate),
                    record_start_ms=start, record_window_ms=1000.0,
                    phase=(formats.PHASE_BASELINE if index == formats.BASELINE_INDEX
                           else formats.PHASE_POST))
            base_times = formats.load_spikes(
                data / formats.probe_filename(formats.SPIKES,
                                              formats.BASELINE_INDEX)).times
            expected = mean_iei_ms(base_times)

            series = _StubSeries(run_dir)
            self.assertAlmostEqual(series.avalanche_bin_ms(), expected, places=9)
            # 2 回目もキャッシュから同じ値
            self.assertAlmostEqual(series.avalanche_bin_ms(), expected, places=9)

    def test_avalanche_bin_needs_a_baseline(self):
        """ベースラインが無ければ測らずに落とすこと (比較の基準が無い)。"""
        from src.utils.runview import MissingData
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(MissingData):
                _StubSeries(Path(tmp_dir)).avalanche_bin_ms()

    def test_probe_numbers_are_record_hours_indices(self):
        """post の番号が `task.record_hours` の添字と一致すること。

        以前はベースラインが 0 を取っていたので、`record_hours: [0..20]` が
        p001..p021 になり「3 番目の記録時刻」と「p003」が食い違っていた。
        ベースラインは番号を持たず `baseline` という名前で区別する。
        """
        from scripts.lesion.store.formats import (BASELINE_INDEX, SPIKES,
                                                  parse_probe_name, probe_filename,
                                                  probe_token)
        self.assertEqual(probe_filename(SPIKES, 0), "spikes_p000.npz")
        self.assertEqual(probe_filename(SPIKES, 20), "spikes_p020.npz")
        self.assertEqual(probe_filename(SPIKES, BASELINE_INDEX), "spikes_baseline.npz")
        # 図のファイル名も npz と同じ token を使う (読み替えなしで突き合わせられる)。
        self.assertEqual(probe_token(3), "p003")
        self.assertEqual(probe_token(BASELINE_INDEX), "baseline")
        # 往復すること。
        for index in (BASELINE_INDEX, 0, 7, 20):
            self.assertEqual(parse_probe_name(probe_filename(SPIKES, index)),
                             (SPIKES, index))

    def test_baseline_sorts_before_the_numbered_probes(self):
        """`discover_probes` が baseline を先頭に置くこと (番号が負なので自然に前)。"""
        from scripts.lesion.store.formats import (BASELINE_INDEX, discover_probes,
                                                  probe_filename)
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            for index in (2, 0, BASELINE_INDEX, 10):
                np.savez_compressed(run_dir / probe_filename("spikes", index),
                                    x=np.array([1]))
            self.assertEqual([p.index for p in discover_probes(run_dir, "spikes")],
                             [BASELINE_INDEX, 0, 2, 10])

    def test_probe_names_do_not_collide_with_akita_records(self):
        """akita_soc の解析 CLI が損傷 run を誤読しないこと。"""
        from scripts.develop.store.formats import RECORD_PATTERN, record_glob
        from scripts.lesion.store.formats import (discover_probes, parse_probe_name,
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
