import math
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
    consume_synaptic_resource,
    e_stdp_kernel,
    i_stdp_kernel,
    recover_synaptic_resource,
)
from src.utils.akita_soc import (
    diagnose_activity,
    spike_group_metrics,
    weight_block_metrics,
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


class AkitaSocMetricsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
