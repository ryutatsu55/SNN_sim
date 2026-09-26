"""刺激境界、欠損指標、閾値校正の意味を検証する。"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.autapse_analysis import (
    matched_dendritic_delay,
    spike_metrics,
    steps_for,
    threshold_bracket,
)


def test_boundaries_and_tail():
    r = spike_metrics([9, 10, 19, 20, 21, 22, 90], 10, 20, 100, 10, 2, 3)
    assert (r["pre_spikes"], r["during_spikes"], r["post_spikes"]) == (1, 2, 4)
    assert r["tail_spikes"] == 1
    assert r["first_post_latency_ms"] == 0
    assert r["last_post_latency_ms"] == 70
    assert r["post_bursts"] == 1
    assert r["status"] == "active_at_observation_tail"


def test_no_spikes_and_single_spike_are_not_zero_isi():
    for spikes in ([], [21]):
        r = spike_metrics(spikes, 10, 20, 100, 10)
        assert r["post_isi_mean_ms"] is None
        assert r["post_isi_cv"] is None
    assert spike_metrics([], 10, 20, 100, 10)["last_post_latency_ms"] is None
    assert (
        spike_metrics([21], 10, 20, 100, 10)["status"]
        == "post_spikes_without_tail_activity"
    )


def test_invalid_times_and_step_rounding():
    assert steps_for(0.3, 0.1) == 3
    for value in (0.15, np.nan, -1):
        with pytest.raises(ValueError):
            steps_for(value, 0.1)
    with pytest.raises(ValueError):
        spike_metrics([100], 10, 20, 100, 10)


def test_threshold_reports_interval_and_rejects_unsupported_comparison():
    assert threshold_bracket([0, 1, 2, 3], [False, False, True, True]) == {
        "lower_nonfiring": 1,
        "upper_firing": 2,
    }
    for fired in ([True, True, True], [False, False, False], [False, True, False]):
        with pytest.raises(ValueError):
            threshold_bracket([0, 1, 2], fired)


def test_refinement_preserves_physical_arrival_delay():
    for dt in (0.1, 0.05, 0.025):
        delay = matched_dendritic_delay(5, dt, 0.1)
        assert delay + dt == pytest.approx(5.1)
    with pytest.raises(ValueError):
        matched_dendritic_delay(5, 0.04, 0.1)
