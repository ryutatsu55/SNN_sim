"""配線の同一性、入力、E/I符号、実効遅延と試行独立性を検証する。"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
from pydantic import ValidationError
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.core.network_snapshot import (
    array_hash,
    arrival_to_dendritic_ms,
    load_snapshot,
    save_snapshot,
)
from src.utils.experiments.baseline_network import (
    BaselineSettings,
    make_app_config,
    make_snapshot,
    run_case,
    summarize_spikes,
)


@pytest.fixture
def cfg():
    values = yaml.safe_load(
        (Path(__file__).parent / "fixtures/baseline_small.yaml").read_text()
    )
    return BaselineSettings.model_validate(values)


def changed(cfg, **changes):
    return BaselineSettings.model_validate({**cfg.model_dump(), **changes})


def test_snapshot_roundtrip_and_space_independence(cfg, tmp_path):
    plain = tmp_path / "plain.npz"
    spatial = tmp_path / "spatial.npz"
    make_snapshot(cfg, 11, plain)
    make_snapshot(
        changed(cfg, space={"kind": "uniform_2d", "extent": 200, "unit": "um"}),
        11,
        spatial,
    )
    a, meta, first_hash = load_snapshot(plain)
    b, _, second_hash = load_snapshot(spatial)
    assert first_hash != second_hash
    assert a["coords"].shape == (cfg.n, 0)
    assert b["coords"].shape == (cfg.n, 2)
    for key in a:
        if key != "coords":
            np.testing.assert_array_equal(a[key], b[key])
    assert not np.any(a["rows"] == a["cols"])
    assert np.all(
        a["rows"] * cfg.n + a["cols"] == np.unique(a["rows"] * cfg.n + a["cols"])
    )
    assert set(a["autapse_exc_order"][:3]) <= set(a["autapse_exc_order"][:6])
    assert set(a["autapse_inh_order"]) == set(range(cfg.n_exc, cfg.n))
    replay = tmp_path / "replay.npz"
    assert save_snapshot(replay, a, meta) == first_hash


def test_corruption_and_self_edges_are_rejected(cfg, tmp_path):
    path = tmp_path / "snapshot.npz"
    make_snapshot(cfg, 11, path)
    arrays, meta, fingerprint = load_snapshot(path)
    arrays["cols"][0] = arrays["rows"][0]
    with pytest.raises(ValueError, match="自己結合"):
        save_snapshot(tmp_path / "bad.npz", arrays, meta)
    arrays, _, _ = load_snapshot(path)
    arrays["arrival_ms"][0] += 1
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            **arrays,
            metadata=np.array(json.dumps(meta, sort_keys=True, ensure_ascii=False)),
            fingerprint=np.array(fingerprint),
        )
    with pytest.raises(ValueError, match="ハッシュ"):
        load_snapshot(path)


def test_invalid_settings_and_delay_resolution(cfg):
    for patch in (
        {"n_exc": cfg.n},
        {"dt_ms": 0.04},
        {"tail_ms": 200},
        {"weights": {"ie": 1}},
        {"space": {"kind": "distance"}},
        {"multipliers": [1, 2]},
        {"multipliers": [0, float("nan")]},
        {"sample_ms": 1.1},
    ):
        with pytest.raises((ValidationError, ValueError)):
            changed(cfg, **patch)
    for dt in (0.05, 0.025):
        delay = arrival_to_dendritic_ms([1.5], dt)
        np.testing.assert_allclose(delay + dt, [1.5])
    with pytest.raises(ValueError):
        arrival_to_dendritic_ms([20], 0.05)


def test_spike_boundaries_and_empty_result(cfg):
    stop = cfg.warmup_ms + cfg.pulse_ms
    metrics, rates = summarize_spikes(
        np.array([99.95, 100, stop, cfg.end_ms - cfg.tail_ms]),
        np.array([0, 0, 1, cfg.n_exc]),
        cfg,
    )
    assert metrics["pre_spikes"] == 1
    assert metrics["during_spikes"] == 1
    assert metrics["post_spikes"] == 2
    assert metrics["tail_spikes"] == 1
    assert metrics["post_participants"] == 2
    metrics, _ = summarize_spikes(np.array([]), np.array([], dtype=int), cfg)
    assert metrics["post_spikes"] == 0
    assert metrics["last_post_latency_ms"] is None
    assert metrics["spectrum_peak_hz"] is None


def test_builder_coo_and_pair_weights(cfg, tmp_path):
    pytest.importorskip("pygenn")
    from src.core.NetworkBuilder import NetworkBuilder
    import src.models.network.snapshot
    import src.models.neurons.pqn_float
    import src.models.plasticity.standard_models
    import src.models.synapses.standard_models

    path = tmp_path / "snapshot.npz"
    make_snapshot(cfg, 11, path)
    arrays, _, _ = load_snapshot(path)
    app = make_app_config(cfg, path, 2, 0.5)
    assert app.network.area.profile_name == "no_space"
    builder = NetworkBuilder(app)
    builder.build()
    codes = []
    for index in builder.synapse_index.values():
        codes.extend(index.global_src * cfg.n + index.global_tgt)
    np.testing.assert_array_equal(
        np.sort(codes), arrays["rows"] * cfg.n + arrays["cols"]
    )
    expected = np.array(
        [cfg.weights.ee * 2, cfg.weights.ei * 2, cfg.weights.ie, cfg.weights.ii]
    )
    pair = (arrays["rows"] >= cfg.n_exc).astype(int) * 2 + (arrays["cols"] >= cfg.n_exc)
    coo = builder.global_coo()
    np.testing.assert_array_equal(coo.row, arrays["rows"])
    np.testing.assert_array_equal(coo.col, arrays["cols"])
    np.testing.assert_allclose(coo.weights, expected[pair])
    np.testing.assert_allclose(coo.delays + cfg.dt_ms, arrays["arrival_ms"])
    with pytest.raises(ValueError, match="N/E/I"):
        make_app_config(changed(cfg, n=21), path, 1, 1)


def test_cpu_input_controls_and_independent_trials(cfg, tmp_path):
    pytest.importorskip("pygenn")
    cfg = changed(cfg, warmup_ms=500, post_ms=100)
    path = tmp_path / "snapshot.npz"
    make_snapshot(cfg, 11, path)
    rows = []
    for name, no_stimulus in (("first", False), ("second", False), ("zero", True)):
        rows.append(
            run_case(
                cfg,
                path,
                tmp_path / name,
                tmp_path / "build" / name,
                multiplier=0,
                no_stimulus=no_stimulus,
            )
        )
    first = np.load(tmp_path / "first/spikes.npz")
    second = np.load(tmp_path / "second/spikes.npz")
    np.testing.assert_array_equal(first["times_ms"], second["times_ms"])
    np.testing.assert_array_equal(first["ids"], second["ids"])
    assert rows[0]["stimulated_during_participants"] == 4
    assert (
        rows[0]["post_spikes"]
        == rows[2]["during_spikes"]
        == rows[2]["post_spikes"]
        == 0
    )
    traces = np.load(tmp_path / "first/traces.npz")
    t = traces["input_time_start_ms"]
    expected = np.where((t >= 500) & (t < 520), cfg.stimulus_amplitude, 0)
    np.testing.assert_allclose(traces["max_abs_input"][:, 0], expected)
    np.testing.assert_allclose(traces["max_abs_input"][:, 1], 0)


@pytest.mark.parametrize("dt", [0.05, 0.025])
def test_builder_real_delay_sign_and_decay(cfg, tmp_path, dt):
    pygenn = pytest.importorskip("pygenn")
    from src.core.NetworkBuilder import NetworkBuilder
    from src.core.registry import NEURON_MODELS
    from src.models.neurons.BASE_neuron import BaseNeuronModel
    import src.models.network.snapshot
    import src.models.plasticity.standard_models
    import src.models.synapses.standard_models

    probe_name = f"baseline_test_probe_{round(1000 * dt)}"

    @NEURON_MODELS.register(probe_name)
    class Probe(BaseNeuronModel):
        @property
        def model_class(self):
            return pygenn.create_neuron_model(
                "BaselineProbe",
                vars=[("Seen", "scalar")],
                sim_code="Seen = Isyn;",
                threshold_condition_code="t >= 1.0 && t < 1.0 + dt * 0.5",
            )

        @property
        def params(self):
            return {}

        @property
        def vars(self):
            return {"Seen": 0.0}

    cfg = changed(cfg, n=2, n_exc=1, probability=1.0, dt_ms=dt)
    path = tmp_path / "snapshot.npz"
    make_snapshot(cfg, 11, path)
    app = make_app_config(cfg, path, 1, 1)
    for cell in app.neurons.values():
        cell.type = probe_name
    builder = NetworkBuilder(app, code_gen_dir=str(tmp_path / "build"))
    model, _ = builder.build()
    (tmp_path / "build").mkdir(exist_ok=True)
    model.build(path_to_model=str(tmp_path / "build"))
    steps = round(5 / dt)
    onset = round(2.5 / dt)
    model.load(num_recording_timesteps=steps)
    observed = []
    try:
        for _ in range(steps):
            model.step_time()
            row = []
            for pop in model.neuron_populations.values():
                pop.vars["Seen"].pull_from_device()
                row.append(pop.vars["Seen"].view.item())
            observed.append(row)
    finally:
        model.unload()
    observed = np.array(observed)
    onsets = np.argmax(np.abs(observed) > 0, axis=0)
    np.testing.assert_array_equal(onsets, [onset, onset])  # 発火1ms + 到着遅延1.5ms
    tau = np.array([cfg.tau_inh_ms, cfg.tau_exc_ms])
    weights = np.array([cfg.weights.ie, cfg.weights.ei])
    expected = weights * -tau * np.expm1(-cfg.dt_ms / tau) / cfg.dt_ms
    np.testing.assert_allclose(observed[onset], expected, rtol=1e-6)
    np.testing.assert_allclose(
        observed[onset + 1] / observed[onset], np.exp(-cfg.dt_ms / tau), rtol=1e-6
    )


@pytest.mark.parametrize(
    "failure", ["build", "persistent_build", "simulation", "numerical"]
)
def test_only_pre_simulation_native_failure_is_retried(tmp_path, monkeypatch, failure):
    import src.utils.experiments.baseline_network as experiment

    calls = []

    class FakeProcess:
        def __init__(self, target, args):
            self.out = Path(args[2])
            self.exitcode = 0

        def start(self):
            calls.append(self.out)
            self.out.mkdir()
            if len(calls) > 1 and failure != "persistent_build":
                (self.out / "summary.json").write_text('{"post_spikes": 0}')
                return
            self.exitcode = -11
            phase = (
                "before_simulation"
                if failure in ("build", "persistent_build")
                else "simulation"
            )
            (self.out / "status.json").write_text(json.dumps({"phase": phase}))
            if failure == "numerical":
                self.exitcode = 1
                (self.out / "worker_error.json").write_text(
                    json.dumps({"type": "FloatingPointError", "error": "non-finite"})
                )

        def join(self):
            pass

    class FakeContext:
        Process = FakeProcess

    monkeypatch.setattr(
        experiment.multiprocessing, "get_context", lambda _: FakeContext()
    )
    out = tmp_path / "trial"
    if failure == "build":
        assert run_case(None, None, out, tmp_path / "build") == {"post_spikes": 0}
        assert len(calls) == 2
        assert (tmp_path / "trial_build_failure_1/status.json").exists()
        assert len(json.loads((out / "build_retries.json").read_text())) == 1
    else:
        expected = FloatingPointError if failure == "numerical" else RuntimeError
        with pytest.raises(expected):
            run_case(None, None, out, tmp_path / "build")
        assert len(calls) == (3 if failure == "persistent_build" else 1)
