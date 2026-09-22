"""Akita SoC のモデルと指標を検算する。

**実験スクリプトのテストではない** (名前が `test_akita_soc.py` だと
`test/experiments/akita_soc/` のものと衝突する)。

対象は**モデルと数式**だけ —— escape LIF の膜電位更新、STDP カーネル、
べき乗フィット、E/I ブロック統計。実験スクリプト (`scripts/akita_soc/`) の
記録規約・図・再解析は `test/experiments/akita_soc/` が見る。
"""
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
    recover_synaptic_resource,
)
from src.utils.analysis.powerlaw import discrete_distribution, fit_distribution_curves
from src.utils.analysis.spikes import diagnose_activity, spike_group_metrics
from src.utils.analysis.weights import (
    block_values,
    compute_block_metrics,
    weight_block_metrics,
)
from src.core.config_manager import ConfigManager
from src.core.layout import NetworkLayout
from src.core.output_manager import AXES_NAME, CONFIG_NAME, locate, require


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
        # 全結合 (4x4 の全ペア) を COO で渡す。結合が無いペアは存在しない構成なので、
        # かつて密行列を丸ごと渡していたケースと同じ統計になる。
        weights = np.zeros((4, 4), dtype=np.float32)
        weights[0, 0] = 1.0
        weights[0, 1] = 0.5
        weights[1, 2] = 1.0
        weights[2, 0] = 0.25
        weights[2, 3] = 1.0
        weights[3, 2] = 0.75
        row, col = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        row, col = row.reshape(-1), col.reshape(-1)

        blocks = block_values(weights.reshape(-1), row, col,
                              minimal_layout(num_exc=2, num_inh=2))
        metrics = weight_block_metrics(blocks, wmax=1.0)

        self.assertAlmostEqual(metrics["weight_mean"], float(np.mean(weights)))
        self.assertAlmostEqual(metrics["weight_at_max_fraction"], 3 / 16)
        self.assertAlmostEqual(metrics["weight_ee_mean"], 0.375)
        self.assertAlmostEqual(metrics["weight_ei_at_max_fraction"], 0.25)
        self.assertAlmostEqual(metrics["weight_ie_mean"], 0.0625)
        self.assertAlmostEqual(metrics["weight_ii_at_max_fraction"], 0.25)

    def test_weight_block_metrics_counts_only_existing_synapses(self):
        """COO は実結合しか持たないので、結合の無いペアの 0 は統計に混ざらない。"""
        row = np.array([0, 1])
        col = np.array([1, 2])
        weights = np.array([1.0, 0.5], dtype=np.float32)

        blocks = block_values(weights, row, col, minimal_layout(num_exc=2, num_inh=1))
        metrics = weight_block_metrics(blocks, wmax=1.0)

        self.assertAlmostEqual(metrics["weight_mean"], 0.75)
        self.assertAlmostEqual(metrics["weight_at_max_fraction"], 0.5)
        self.assertAlmostEqual(metrics["weight_ei_mean"], 0.5)

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
  area:
    profile_name: no_space
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


if __name__ == "__main__":
    unittest.main()
