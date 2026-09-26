"""CPU実測でパルスの境界と遅延シナプスの到着時刻を検証する。"""

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
pygenn = pytest.importorskip("pygenn")
from scripts.autapse_comparison import Case, ROOT, delayed_pulse_model, run_bank


def test_rectangular_and_exponential_input(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/autapse_comparison.yaml").read_text())
    cfg.update(warmup_ms=10.0, post_ms=20.0, tail_ms=10.0)
    rows = run_bank(
        "TestInputs",
        cfg["models"]["LIF"],
        cfg,
        [
            Case("zero", "rect", 0, 20),
            Case("rect", "rect", 1, 20),
            Case("exp", "exp", 1, 40),
        ],
        tmp_path / "out",
        tmp_path / "build",
    )
    data = np.load(tmp_path / "out/traces.npz")
    t, actual = data["time_start_ms"], data["injected"]
    assert np.all(actual[:, 0] == 0)
    np.testing.assert_allclose(actual[:, 1], ((t >= 10) & (t < 30)).astype(float))
    expected = np.where(
        (t >= 10) & (t < 50), -2 * np.expm1(-0.1 / 2) / 0.1 * np.exp(-(t - 10) / 2), 0
    )
    np.testing.assert_allclose(actual[:, 2], expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(data["V"][:, 0], -60)
    assert rows[0]["during_spikes"] == rows[0]["post_spikes"] == 0
    assert rows[1]["during_spikes"] == 1
    # LIFの解析解：20*log(2) ms後のステップで最初の発火。
    spikes = np.loadtxt(tmp_path / "out/spikes.csv", delimiter=",", skiprows=1, ndmin=2)
    first = spikes[spikes[:, 1] == 1, 0][0]
    assert abs(first - (10 + 20 * np.log(2))) <= cfg["dt_ms"]


def test_dendritic_delay_and_expcurr_gain(tmp_path):
    model = pygenn.GeNNModel(
        "float", "TestDelay", backend="single_threaded_cpu", time_precision="double"
    )
    model.dt = 0.1
    source = model.add_neuron_population(
        "Source", 1, "SpikeSourceArray", {}, {"startSpike": 0, "endSpike": 1}
    )
    source.extra_global_params["spikeTimes"].set_init_values(np.array([1.0]))
    probe_model = pygenn.create_neuron_model(
        "Probe", vars=[("Seen", "scalar")], sim_code="Seen = Isyn;"
    )
    probe = model.add_neuron_population("Probe", 3, probe_model, {}, {"Seen": 0.0})
    delayed = delayed_pulse_model()
    syn = model.add_synapse_population(
        "Delayed",
        "SPARSE",
        source,
        probe,
        pygenn.init_weight_update(
            delayed, {}, {"g": np.ones(3), "d": np.array([0, 10, 300], dtype=np.uint32)}
        ),
        pygenn.init_postsynaptic("ExpCurr", {"tau": 2.0}, {}),
    )
    syn.set_sparse_connections(
        np.zeros(3, dtype=np.uint32), np.arange(3, dtype=np.uint32)
    )
    syn.max_dendritic_delay_timesteps = 301
    model.build(path_to_model=str(tmp_path))
    model.load()
    observed = []
    try:
        for _ in range(400):
            model.step_time()
            probe.vars["Seen"].pull_from_device()
            observed.append(probe.vars["Seen"].view.copy().reshape(-1))
    finally:
        model.unload()
    observed = np.array(observed)
    onset_steps = np.argmax(observed > 0, axis=0)
    np.testing.assert_array_equal(onset_steps, [11, 21, 311])
    np.testing.assert_allclose(
        observed[onset_steps, np.arange(3)], -2 * np.expm1(-0.1 / 2) / 0.1, rtol=1e-5
    )
    np.testing.assert_allclose(
        observed[onset_steps + 1, np.arange(3)] / observed[onset_steps, np.arange(3)],
        np.exp(-0.1 / 2),
        rtol=1e-5,
    )
