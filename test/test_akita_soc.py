import math
import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np
import yaml

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.models.neurons.akita_escape_lif import (
    calculate_escape_noise_scale,
    conductance_lif_delta,
    conductance_lif_delta_from_conductances,
    conductance_synaptic_current,
    escape_noise_probability,
    evolve_escape_lif_step,
)
from src.models.plasticity.custom_Akita import (
    calculate_gmax_scale,
    consume_synaptic_resource,
    decay_trace,
    e_stdp_kernel,
    e_trace_post_delta,
    e_trace_pre_delta,
    i_stdp_kernel,
    i_trace_delta,
    recover_synaptic_resource,
)
from src.utils.analysis.powerlaw import discrete_distribution, fit_distribution_curves
from src.utils.analysis.spikes import diagnose_activity, spike_group_metrics
from src.utils.analysis.weights import (
    block_values,
    block_values_coo,
    compute_block_metrics,
    weight_block_metrics,
)
from src.utils.plotting.distributions import plot_avalanche_distribution
from src.utils.plotting.raster import plot_raster
from scripts.akita_soc_fig2 import replot_existing_output
from src.core.config_manager import ConfigManager
from src.core.layout import NetworkLayout
from src.core.output_manager import AXES_NAME, CONFIG_NAME, locate, require
from src.utils.experiments.akita_soc.runio import (
    SPIKES,
    WEIGHTS,
    discover_records,
    parse_hour,
    record_filename,
)
from src.utils.experiments.akita_soc.weight_track import visualize_weight_tracks


class AkitaEscapeLIFTest(unittest.TestCase):
    def test_conductance_lif_delta_matches_equation(self):
        delta = conductance_lif_delta(v=-70.0, isyn=5.0, dt=0.1, tau_m=30.0, v_rest=-74.0)
        expected = (0.1 / 30.0) * ((-74.0 + 70.0) + 5.0)
        self.assertAlmostEqual(delta, expected)

    def test_conductance_terms_match_supplementary_equation(self):
        isyn = conductance_synaptic_current(
            v=-70.0,
            g_exc=0.2,
            g_inh=0.1,
            e_exc=0.0,
            e_inh=-80.0,
        )
        self.assertAlmostEqual(isyn, ((0.0 + 70.0) * 0.2) + ((-80.0 + 70.0) * 0.1))

        delta = conductance_lif_delta_from_conductances(
            v=-70.0,
            g_exc=0.2,
            g_inh=0.1,
            dt=0.1,
            tau_m=30.0,
            v_rest=-74.0,
            e_exc=0.0,
            e_inh=-80.0,
        )
        expected = (0.1 / 30.0) * ((-74.0 + 70.0) + isyn)
        self.assertAlmostEqual(delta, expected)

    def test_escape_noise_probability_increases_with_voltage(self):
        scale = calculate_escape_noise_scale(dt=0.1, frest=0.4, v_rest=-74.0, v_th=-54.0, b=4.0)
        low = escape_noise_probability(v=-70.0, v_th=-54.0, b=4.0, scale_c=scale)
        high = escape_noise_probability(v=-55.0, v_th=-54.0, b=4.0, scale_c=scale)
        self.assertLess(low, high)

    def test_refractory_state_blocks_spike(self):
        next_v, next_ref, spiked, prob = evolve_escape_lif_step(
            v=-74.0,
            refrac_time=2.0,
            isyn=100.0,
            i_ext=0.0,
            dt=0.1,
            tau_m=30.0,
            v_rest=-74.0,
            v_th=-54.0,
            b=4.0,
            scale_c=1.0,
            tau_refrac=3.0,
            random_uniform=0.0,
        )
        self.assertFalse(spiked)
        self.assertEqual(prob, 0.0)
        self.assertAlmostEqual(next_v, -74.0)
        self.assertAlmostEqual(next_ref, 1.9)


class AkitaPlasticityTest(unittest.TestCase):
    def test_gmax_scale_defaults_to_unscaled(self):
        self.assertAlmostEqual(
            calculate_gmax_scale(num_synapses=6320, num_post=80, normalize_by_fan_in=False),
            1.0,
        )

    def test_gmax_scale_uses_average_fan_in(self):
        self.assertAlmostEqual(
            calculate_gmax_scale(num_synapses=6320, num_post=80, normalize_by_fan_in=True),
            1.0 / 79.0,
        )
        self.assertAlmostEqual(
            calculate_gmax_scale(num_synapses=1600, num_post=20, normalize_by_fan_in=True),
            1.0 / 80.0,
        )

    def test_gmax_scale_handles_empty_connections(self):
        self.assertAlmostEqual(
            calculate_gmax_scale(num_synapses=0, num_post=20, normalize_by_fan_in=True),
            1.0,
        )

    def test_stp_recovers_then_depletes(self):
        recovered = recover_synaptic_resource(x=0.2, delta_t=150.0, tau_rec=150.0)
        remaining, released = consume_synaptic_resource(recovered, utilization=0.4)
        self.assertGreater(recovered, 0.2)
        self.assertAlmostEqual(remaining + released, recovered)

    def test_e_stdp_kernel_has_expected_signs(self):
        potentiate = e_stdp_kernel(delta_t=10.0, a_e=0.02, tau_e=20.0, beta_e=1.0)
        depress = e_stdp_kernel(delta_t=-10.0, a_e=0.02, tau_e=20.0, beta_e=1.15)
        self.assertGreater(potentiate, 0.0)
        self.assertLess(depress, 0.0)

    def test_i_stdp_kernel_is_symmetric(self):
        positive = i_stdp_kernel(delta_t=8.0, a_i=0.02, tau_i1=10.0, tau_i2=20.0, beta_i=1.15)
        negative = i_stdp_kernel(delta_t=-8.0, a_i=0.02, tau_i1=10.0, tau_i2=20.0, beta_i=1.15)
        self.assertAlmostEqual(positive, negative)

    def test_trace_decay_matches_exponential_sum_term(self):
        self.assertAlmostEqual(decay_trace(trace=2.0, elapsed=20.0, tau=20.0), 2.0 * math.exp(-1.0))
        self.assertAlmostEqual(decay_trace(trace=2.0, elapsed=0.0, tau=20.0), 2.0)

    def test_e_trace_deltas_match_accumulated_stdp_window(self):
        pre_trace = math.exp(-10.0 / 20.0) + math.exp(-30.0 / 20.0)
        post_trace = math.exp(-5.0 / 20.0) + math.exp(-25.0 / 20.0)

        self.assertAlmostEqual(e_trace_post_delta(pre_trace, a_e=0.02), 0.02 * pre_trace)
        self.assertAlmostEqual(e_trace_pre_delta(post_trace, a_e=0.02, beta_e=1.0), -0.02 * post_trace)

    def test_i_trace_delta_matches_accumulated_symmetric_window(self):
        tau_i1 = 10.0
        tau_i2 = 20.0
        beta_i = 1.15
        trace1 = math.exp(-8.0 / tau_i1) + math.exp(-18.0 / tau_i1)
        trace2 = math.exp(-8.0 / tau_i2) + math.exp(-18.0 / tau_i2)
        expected = i_stdp_kernel(8.0, 0.02, tau_i1, tau_i2, beta_i)
        expected += i_stdp_kernel(18.0, 0.02, tau_i1, tau_i2, beta_i)

        self.assertAlmostEqual(i_trace_delta(trace1, trace2, 0.02, tau_i1, tau_i2, beta_i), expected)


class AkitaSocMetricsTest(unittest.TestCase):
    def test_discrete_distribution_can_include_sizes_above_fitting_limit(self):
        sizes = np.array([1, 2, 100, 101, 150], dtype=np.int32)

        support, prob = discrete_distribution(sizes, xmax=None)

        self.assertTrue(np.array_equal(support, np.array([1, 2, 100, 101, 150])))
        self.assertTrue(np.allclose(prob, np.full(5, 0.2)))

    def test_discrete_distribution_keeps_explicit_fitting_limit(self):
        sizes = np.array([1, 2, 100, 101, 150], dtype=np.int32)

        support, prob = discrete_distribution(sizes, xmax=100)

        self.assertTrue(np.array_equal(support, np.array([1, 2, 100])))
        self.assertTrue(np.allclose(prob, np.full(3, 1 / 3)))

    def test_fit_distribution_curves_scales_to_empirical_mass(self):
        # fit_max を超えるサイズを混ぜると、理論曲線は [1, fit_max] の経験質量に合わせて
        # 縮む (経験 PMF と重ね描きできるようにするため)。
        rng = np.random.default_rng(0)
        sizes = np.concatenate([rng.integers(1, 40, size=500), np.array([120, 300])])

        fit = fit_distribution_curves(sizes, fit_max=100)

        mass = float(fit.prob[fit.support <= 100].sum())
        self.assertLess(mass, 1.0)
        self.assertAlmostEqual(float(fit.powerlaw.sum()), mass)
        self.assertAlmostEqual(float(fit.exponential.sum()), mass)
        self.assertEqual(fit.fit_support.size, 100)
        self.assertEqual(fit.num_fitted, 500)

    def test_fit_distribution_curves_reports_no_curve_when_underdetermined(self):
        for sizes in (np.array([], dtype=np.int64), np.array([5])):
            fit = fit_distribution_curves(sizes, fit_max=100)
            self.assertEqual(fit.powerlaw.size, 0)
            self.assertEqual(fit.exponential.size, 0)
            self.assertTrue(np.isnan(fit.llr))

    def test_spike_group_metrics_uses_global_group_ids(self):
        spike_ids = np.array([2, 5, 5, 7, 9, 9, 9])
        excitatory_ids = np.array([5, 9])
        inhibitory_ids = np.array([2, 7])

        metrics = spike_group_metrics(
            spike_ids=spike_ids,
            excitatory_ids=excitatory_ids,
            inhibitory_ids=inhibitory_ids,
            duration_ms=1000.0,
        )

        self.assertEqual(metrics["exc_spikes"], 5)
        self.assertEqual(metrics["inh_spikes"], 2)
        self.assertAlmostEqual(metrics["exc_rate_hz"], 2.5)
        self.assertAlmostEqual(metrics["inh_rate_hz"], 1.0)

    def test_weight_block_metrics_reports_block_saturation(self):
        weights = np.zeros((4, 4), dtype=np.float32)
        weights[0, 0] = 1.0
        weights[0, 1] = 0.5
        weights[1, 2] = 1.0
        weights[2, 0] = 0.25
        weights[2, 3] = 1.0
        weights[3, 2] = 0.75

        blocks = block_values(weights, minimal_layout(num_exc=2, num_inh=2))
        metrics = weight_block_metrics(blocks, wmax=1.0)

        self.assertAlmostEqual(metrics["weight_mean"], float(np.mean(weights)))
        self.assertAlmostEqual(metrics["weight_at_max_fraction"], 3 / 16)
        self.assertAlmostEqual(metrics["weight_ee_mean"], 0.375)
        self.assertAlmostEqual(metrics["weight_ei_at_max_fraction"], 0.25)
        self.assertAlmostEqual(metrics["weight_ie_mean"], 0.0625)
        self.assertAlmostEqual(metrics["weight_ii_at_max_fraction"], 0.25)

    def test_weight_block_metrics_can_use_connection_mask(self):
        weights = np.zeros((3, 3), dtype=np.float32)
        weights[0, 1] = 1.0
        weights[1, 2] = 0.5
        mask = np.zeros((3, 3), dtype=np.int32)
        mask[0, 1] = 1
        mask[1, 2] = 1

        blocks = block_values(weights, minimal_layout(num_exc=2, num_inh=1), connection_mask=mask)
        metrics = weight_block_metrics(blocks, wmax=1.0)

        self.assertAlmostEqual(metrics["weight_mean"], 0.75)
        self.assertAlmostEqual(metrics["weight_at_max_fraction"], 0.5)
        self.assertAlmostEqual(metrics["weight_ei_mean"], 0.5)

    def test_dense_and_coo_block_decomposition_agree(self):
        rng = np.random.default_rng(0)
        layout = minimal_layout(num_exc=3, num_inh=2)
        total = layout.total_neurons
        weights = rng.random((total, total))
        mask = rng.random((total, total)) < 0.6
        np.fill_diagonal(mask, False)
        weights[~mask] = 0.0

        row, col = np.nonzero(mask)
        dense = block_values(weights, layout, connection_mask=mask)
        coo = block_values_coo(weights[row, col], row, col, layout)

        self.assertEqual(set(dense), set(coo))
        for name in dense:
            # COO は行優先、密は np.ix_ の順なので、集合として一致すればよい。
            np.testing.assert_allclose(np.sort(dense[name]), np.sort(coo[name]))
        self.assertEqual(
            weight_block_metrics(dense, wmax=1.0), weight_block_metrics(coo, wmax=1.0)
        )

    def test_diagnose_activity_combines_overactivity_and_saturation(self):
        diagnosis = diagnose_activity(mean_rate_hz=101.0, weight_at_max_fraction=0.88)

        self.assertTrue(diagnosis["is_overactive"])
        self.assertTrue(diagnosis["is_weight_saturated"])
        self.assertEqual(diagnosis["diagnosis"], "overactive_and_weight_saturated")


# 最小構成。sequential 割当なので興奮性が先頭に連番で並ぶ
# (既定の 2/2 なら興奮性=[0,1] / 抑制性=[2,3])。
MINIMAL_RUN_CONFIG = """
simulation:
  N: {total}
  dt: 0.1
  seed: 1
layout:
  assignment: sequential
inputs:
  GaussianNoise:
    enable: false
neurons:
  Exc:
    type: akita_escape_lif
    mode: excitatory
    polarity: excitatory
    num: {num_exc}
  Inh:
    type: akita_escape_lif
    mode: inhibitory
    polarity: inhibitory
    num: {num_inh}
synapses: {{}}
network:
  space:
    profile_name: no_space
  connection:
    profile_name: constant_prob_full
    p: 1.0
    allow_self_connections: false
  weight:
    profile_name: constant_zero
  delay:
    profile_name: constant
task:
  profile_name: test
meta:
  timestamp: test
"""


def write_minimal_run_config(run_dir: Path, num_exc: int = 2, num_inh: int = 2) -> Path:
    """最小構成の resolved config.yaml を run_dir に書き出す。"""
    config_path = run_dir / CONFIG_NAME
    config_path.write_text(
        MINIMAL_RUN_CONFIG.format(total=num_exc + num_inh, num_exc=num_exc, num_inh=num_inh),
        encoding="utf-8",
    )
    return config_path


def minimal_layout(num_exc: int = 2, num_inh: int = 2):
    """E/I だけを持つ最小の NetworkLayout を作る (ブロック分解のテスト用)。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        run_dir = Path(tmp_dir)
        write_minimal_run_config(run_dir, num_exc=num_exc, num_inh=num_inh)
        return load_run_layout(run_dir)


def load_run_layout(run_dir: Path):
    """保存物から NetworkLayout を復元する (各 main() が行う手続きと同じ)。"""
    config = ConfigManager().load_resolved(require(run_dir, CONFIG_NAME))
    layout = NetworkLayout.from_config(config)
    axes_path = locate(run_dir, AXES_NAME)
    if axes_path is not None:
        layout.load_axes_file(axes_path)
    return layout


class AkitaWeightMatrixVisualizationTest(unittest.TestCase):
    def test_parse_hour_reads_both_record_kinds(self):
        self.assertEqual(parse_hour(Path("weights_0h.npz"), WEIGHTS), 0.0)
        self.assertEqual(parse_hour(Path("weights_6h.npz"), WEIGHTS), 6.0)
        self.assertEqual(parse_hour(Path("weights_72h.npz"), WEIGHTS), 72.0)
        self.assertEqual(parse_hour(Path("spikes_1.5h.npz"), SPIKES), 1.5)

    def test_parse_hour_rejects_wrong_kind_and_bad_names(self):
        with self.assertRaises(ValueError):
            parse_hour(Path("spikes_6h.npz"), WEIGHTS)
        for name in ("weights_6.npz", "weights.npz", "connectivity.npz", "weights_xh.npz"):
            with self.assertRaises(ValueError):
                parse_hour(Path(name))

    def test_record_filename_round_trips_through_parse_hour(self):
        for hour in (0.0, 0.5, 6.0, 72.0, 1.25):
            name = record_filename(WEIGHTS, hour)
            self.assertEqual(parse_hour(Path(name), WEIGHTS), hour)

    def test_discover_weight_files_sorts_by_hour(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            for name in ("weights_72h.npz", "weights_0h.npz", "weights_6h.npz"):
                np.savez_compressed(run_dir / name, weights=np.zeros((4, 4), dtype=np.float32))

            discovered = discover_records(run_dir, WEIGHTS)

            self.assertEqual([item.hour for item in discovered], [0.0, 6.0, 72.0])

    def test_compute_block_metrics_reports_all_blocks(self):
        weights = np.array(
            [
                [0.0, 1.0, 0.5, 0.0],
                [0.2, 0.0, 0.7, 0.0],
                [0.1, 0.3, 0.0, 1.0],
                [0.0, 0.4, 0.6, 0.0],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            write_minimal_run_config(run_dir)
            layout = load_run_layout(run_dir)

        ids = layout.ids_by("polarity")
        self.assertEqual(ids["excitatory"].tolist(), [0, 1])
        self.assertEqual(ids["inhibitory"].tolist(), [2, 3])

        rows = compute_block_metrics(hour=6.0, weights=weights, layout=layout)
        by_block = {row["block"]: row for row in rows}

        self.assertEqual(set(by_block), {"all", "ee", "ei", "ie", "ii"})
        self.assertAlmostEqual(by_block["ee"]["mean"], 0.3)
        self.assertAlmostEqual(by_block["ei"]["mean"], 0.3)
        self.assertAlmostEqual(by_block["ie"]["mean"], 0.2)
        self.assertAlmostEqual(by_block["ii"]["mean"], 0.4)

    def test_visualize_run_generates_weight_matrix_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            write_minimal_run_config(run_dir)
            np.savez_compressed(run_dir / "weights_0h.npz", weights=np.zeros((4, 4), dtype=np.float32))
            np.savez_compressed(run_dir / "weights_6h.npz", weights=np.ones((4, 4), dtype=np.float32))

            out_dir = visualize_weight_tracks(run_dir, load_run_layout(run_dir))

            self.assertTrue((out_dir / "weight_matrix_0h.png").exists())
            self.assertTrue((out_dir / "weight_matrix_6h.png").exists())
            self.assertTrue((out_dir / "weight_matrix_panel.png").exists())
            self.assertTrue((out_dir / "weight_delta_panel.png").exists())
            self.assertTrue((out_dir / "weight_block_metrics.csv").exists())
            self.assertFalse((out_dir / "weight_matrix_report.md").exists())


class AkitaSocPlotTest(unittest.TestCase):
    def test_plot_raster_accepts_paper_axis_ranges(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "raster.png"

            plot_raster(
                times=np.array([0.0, 1000.0, 29000.0, 31000.0]),
                ids=np.array([0, 20, 99, 10]),
                out_path=out_path,
                title="Raster",
                xlim_s=(0.0, 30.0),
                ylim_neuron=(0.0, 100.0),
            )

            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


class AkitaSocReplotTest(unittest.TestCase):
    def test_discover_spike_files_sorts_by_hour(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            for name in ("spikes_72h.npz", "spikes_0h.npz", "spikes_6h.npz"):
                np.savez_compressed(run_dir / name, times=np.array([]), ids=np.array([]))

            discovered = discover_records(run_dir, SPIKES)

            self.assertEqual([item.hour for item in discovered], [0.0, 6.0, 72.0])

    def test_replot_existing_output_generates_plots_without_simulation(self):
        source_config = root_path / "outputs" / "akita_soc_72h" / "20260525-180915" / "config.yaml"
        if not source_config.exists():
            self.skipTest("既存のAkita出力config.yamlがありません。")

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            # polarity 軸 / source のリスト必須化 / layout.assignment の実値保存より前に
            # 保存された config なので、コピーする際に現行スキーマへ寄せる。このテストの
            # 主題は replot であって旧 config の読み込みではない。
            saved = yaml.safe_load(source_config.read_text(encoding="utf-8"))
            for n_cfg in saved["neurons"].values():
                n_cfg.setdefault("polarity", n_cfg["mode"])
            for s_cfg in (saved.get("synapses") or {}).values():
                if isinstance(s_cfg.get("source"), str):
                    s_cfg["source"] = [s_cfg["source"]]
            saved.setdefault("layout", {"assignment": "sequential"})
            (run_dir / "config.yaml").write_text(
                yaml.safe_dump(saved, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            np.savez_compressed(
                run_dir / "spikes_0h.npz",
                times=np.array([0.0, 1000.0, 29000.0, 31000.0]),
                ids=np.array([0, 20, 99, 10]),
            )

            replot_existing_output(run_dir)

            self.assertTrue((run_dir / "raster_0h.png").exists())
            self.assertTrue((run_dir / "avalanche_0h.png").exists())
            self.assertTrue((run_dir / "metrics_replot.csv").exists())
            self.assertTrue((run_dir / "spikes_0h.npz").exists())

    def test_plot_avalanche_accepts_paper_axis_ranges_with_empty_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "avalanche.png"

            plot_avalanche_distribution(
                sizes=np.array([], dtype=np.int32),
                out_path=out_path,
                title="Avalanche",
                xlim=(1.0, 1000.0),
                ylim=(1e-5, 1.0),
            )

            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
