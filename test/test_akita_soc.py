import math
import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np

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
    is_trace_mode,
    recover_synaptic_resource,
)
from src.utils.akita_soc import (
    avalanche_distribution,
    diagnose_activity,
    plot_avalanche_distribution,
    plot_raster,
    spike_group_metrics,
    weight_block_metrics,
)
from scripts.akita_soc_fig2 import discover_spike_files, parse_hour_from_spike_path, replot_existing_output
from scripts.visualize_weight_matrix import (
    compute_block_metrics,
    discover_weight_files,
    infer_group_ids,
    parse_hour_from_weight_path,
    visualize_run,
)


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

    def test_trace_mode_recognizes_trace_profiles_with_suffixes(self):
        self.assertTrue(is_trace_mode("e-stdp_akita_fig2_trace"))
        self.assertTrue(is_trace_mode("i-stdp_akita_fig2_trace"))
        self.assertTrue(is_trace_mode("e-stdp_akita_fig2_trace_unscaled_gmax"))
        self.assertTrue(is_trace_mode("i-stdp_akita_fig2_trace_unscaled_gmax"))

    def test_trace_mode_keeps_nearest_neighbor_profiles_disabled(self):
        self.assertFalse(is_trace_mode("e-stdp_akita_fig2_legacy"))
        self.assertFalse(is_trace_mode("i-stdp_akita_fig2_legacy"))
        self.assertFalse(is_trace_mode("e-stdp"))
        self.assertFalse(is_trace_mode("i-stdp"))


class AkitaSocMetricsTest(unittest.TestCase):
    def test_avalanche_distribution_can_include_sizes_above_fitting_limit(self):
        sizes = np.array([1, 2, 100, 101, 150], dtype=np.int32)

        support, prob = avalanche_distribution(sizes, smax=None)

        self.assertTrue(np.array_equal(support, np.array([1, 2, 100, 101, 150])))
        self.assertTrue(np.allclose(prob, np.full(5, 0.2)))

    def test_avalanche_distribution_keeps_explicit_fitting_limit(self):
        sizes = np.array([1, 2, 100, 101, 150], dtype=np.int32)

        support, prob = avalanche_distribution(sizes, smax=100)

        self.assertTrue(np.array_equal(support, np.array([1, 2, 100])))
        self.assertTrue(np.allclose(prob, np.full(3, 1 / 3)))

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

        metrics = weight_block_metrics(
            weights=weights,
            excitatory_ids=np.array([0, 1]),
            inhibitory_ids=np.array([2, 3]),
            wmax=1.0,
        )

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

        metrics = weight_block_metrics(
            weights=weights,
            excitatory_ids=np.array([0, 1]),
            inhibitory_ids=np.array([2]),
            wmax=1.0,
            connection_mask=mask,
        )

        self.assertAlmostEqual(metrics["weight_mean"], 0.75)
        self.assertAlmostEqual(metrics["weight_at_max_fraction"], 0.5)
        self.assertAlmostEqual(metrics["weight_ei_mean"], 0.5)

    def test_diagnose_activity_combines_overactivity_and_saturation(self):
        diagnosis = diagnose_activity(mean_rate_hz=101.0, weight_at_max_fraction=0.88)

        self.assertTrue(diagnosis["is_overactive"])
        self.assertTrue(diagnosis["is_weight_saturated"])
        self.assertEqual(diagnosis["diagnosis"], "overactive_and_weight_saturated")


class AkitaWeightMatrixVisualizationTest(unittest.TestCase):
    def test_parse_hour_from_weight_path(self):
        self.assertEqual(parse_hour_from_weight_path(Path("weights_0h.npz")), 0.0)
        self.assertEqual(parse_hour_from_weight_path(Path("weights_6h.npz")), 6.0)
        self.assertEqual(parse_hour_from_weight_path(Path("weights_72h.npz")), 72.0)

    def test_discover_weight_files_sorts_by_hour(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            for name in ("weights_72h.npz", "weights_0h.npz", "weights_6h.npz"):
                np.savez_compressed(run_dir / name, weights=np.zeros((4, 4), dtype=np.float32))

            discovered = discover_weight_files(run_dir)

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
        group_ids = infer_group_ids(Path("missing"), matrix_size=4)
        group_ids = type(group_ids)(
            excitatory=np.array([0, 1], dtype=np.int32),
            inhibitory=np.array([2, 3], dtype=np.int32),
            total_neurons=4,
            source="test",
        )

        rows = compute_block_metrics(hour=6.0, weights=weights, group_ids=group_ids)
        by_block = {row["block"]: row for row in rows}

        self.assertEqual(set(by_block), {"all", "ee", "ei", "ie", "ii"})
        self.assertAlmostEqual(by_block["ee"]["mean"], 0.3)
        self.assertAlmostEqual(by_block["ei"]["mean"], 0.3)
        self.assertAlmostEqual(by_block["ie"]["mean"], 0.2)
        self.assertAlmostEqual(by_block["ii"]["mean"], 0.4)

    def test_visualize_run_generates_weight_matrix_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            config = """
simulation:
  N: 4
  dt: 0.1
  seed: 1
inputs:
  GaussianNoise:
    enable: false
neurons:
  Exc:
    type: akita_escape_lif
    mode: excitatory
    num: 2
  Inh:
    type: akita_escape_lif
    mode: inhibitory
    num: 2
synapses: {}
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
            (run_dir / "config.yaml").write_text(config, encoding="utf-8")
            np.savez_compressed(run_dir / "weights_0h.npz", weights=np.zeros((4, 4), dtype=np.float32))
            np.savez_compressed(run_dir / "weights_6h.npz", weights=np.ones((4, 4), dtype=np.float32))

            out_dir = visualize_run(run_dir)

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
    def test_parse_hour_from_spike_path(self):
        self.assertEqual(parse_hour_from_spike_path(Path("spikes_0h.npz")), 0.0)
        self.assertEqual(parse_hour_from_spike_path(Path("spikes_1.5h.npz")), 1.5)
        self.assertEqual(parse_hour_from_spike_path(Path("spikes_72h.npz")), 72.0)

    def test_discover_spike_files_sorts_by_hour(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            for name in ("spikes_72h.npz", "spikes_0h.npz", "spikes_6h.npz"):
                np.savez_compressed(run_dir / name, times=np.array([]), ids=np.array([]))

            discovered = discover_spike_files(run_dir)

            self.assertEqual([hour for hour, _ in discovered], [0.0, 6.0, 72.0])

    def test_replot_existing_output_generates_plots_without_simulation(self):
        source_config = root_path / "outputs" / "akita_soc_72h" / "20260525-180915" / "config.yaml"
        if not source_config.exists():
            self.skipTest("既存のAkita出力config.yamlがありません。")

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            (run_dir / "config.yaml").write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
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
