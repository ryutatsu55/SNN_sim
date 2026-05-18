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
    escape_noise_probability,
    evolve_escape_lif_step,
)
from src.models.plasticity.custom_Akita import (
    consume_synaptic_resource,
    e_stdp_kernel,
    i_stdp_kernel,
    recover_synaptic_resource,
)


class AkitaEscapeLIFTest(unittest.TestCase):
    def test_conductance_lif_delta_matches_equation(self):
        delta = conductance_lif_delta(v=-70.0, isyn=5.0, dt=0.1, tau_m=30.0, v_rest=-74.0)
        expected = (0.1 / 30.0) * ((-74.0 + 70.0) + 5.0)
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



if __name__ == "__main__":
    unittest.main()
