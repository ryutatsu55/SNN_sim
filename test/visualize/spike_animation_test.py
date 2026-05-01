import sys
from pathlib import Path

import numpy as np

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.utils.visualize.spike_animation import _compute_decay_intensity  # noqa: E402


def test_compute_decay_intensity_uses_exponential_decay():
    sorted_time = np.array([10.0, 15.0], dtype=float)
    sorted_id = np.array([0, 1], dtype=int)
    display_index = {0: 0, 1: 1}

    intensity = _compute_decay_intensity(
        frame_time=20.0,
        sorted_time=sorted_time,
        sorted_id=sorted_id,
        display_index=display_index,
        num_display_ids=2,
        decay_tau_ms=10.0,
        decay_cutoff_ms=50.0,
    )

    expected = np.array([np.exp(-1.0), np.exp(-0.5)], dtype=float)
    np.testing.assert_allclose(intensity, expected)


def test_compute_decay_intensity_keeps_max_for_same_neuron():
    sorted_time = np.array([10.0, 18.0], dtype=float)
    sorted_id = np.array([0, 0], dtype=int)
    display_index = {0: 0}

    intensity = _compute_decay_intensity(
        frame_time=20.0,
        sorted_time=sorted_time,
        sorted_id=sorted_id,
        display_index=display_index,
        num_display_ids=1,
        decay_tau_ms=10.0,
        decay_cutoff_ms=50.0,
    )

    np.testing.assert_allclose(intensity, np.array([np.exp(-0.2)], dtype=float))
