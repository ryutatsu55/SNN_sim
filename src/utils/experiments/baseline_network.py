"""PQN基準探索の設定、構造生成、既存Builderによる実行と記録。"""

import csv
import faulthandler
import hashlib
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from src.core.network_snapshot import (
    array_hash,
    arrival_to_dendritic_ms,
    load_snapshot,
    save_snapshot,
)
from src.utils.autapse_analysis import steps_for

ROOT = Path(__file__).resolve().parents[3]


class StrictSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SpaceSettings(StrictSettings):
    kind: Literal["none", "uniform_2d"] = "none"
    seed: int = Field(default=71, ge=0)
    extent: float = Field(default=1.0, gt=0)
    unit: str = "model_unit"
    boundary: Literal["open"] = "open"


class CellSettings(StrictSettings):
    type: Literal["PQN_float"] = "PQN_float"
    mode: Literal["RSexci", "RSinhi"]


class PairWeights(StrictSettings):
    ee: float = Field(default=0.1, ge=0)
    ei: float = Field(default=0.04, ge=0)
    ie: float = Field(default=-0.4, le=0)
    ii: float = Field(default=-0.16, le=0)


class BaselineSettings(StrictSettings):
    n: int = Field(default=1000, ge=2)
    n_exc: int = Field(default=800, ge=1)
    connection_rule: Literal["constant_probability"] = "constant_probability"
    probability: float = Field(default=0.1, ge=0, le=1)
    wiring_seeds: list[int] = Field(default_factory=lambda: [11, 22, 33], min_length=1)
    stimulus_seeds: list[int] = Field(
        default_factory=lambda: [101, 202, 303], min_length=1
    )
    selection_seed: int = Field(default=17, ge=0)
    space: SpaceSettings = Field(default_factory=SpaceSettings)
    delay_rule: Literal["constant_arrival"] = "constant_arrival"
    arrival_ms: float = Field(default=1.5, gt=0)
    dt_ms: float = Field(default=0.05, gt=0)
    backend: Literal["cpu", "cuda"] = "cpu"
    exc_model: CellSettings = Field(default_factory=lambda: CellSettings(mode="RSexci"))
    inh_model: CellSettings = Field(default_factory=lambda: CellSettings(mode="RSinhi"))
    tau_exc_ms: float = Field(default=2.0, gt=0)
    tau_inh_ms: float = Field(default=5.0, gt=0)
    weights: PairWeights = Field(default_factory=PairWeights)
    multipliers: list[float] = Field(
        default_factory=lambda: [0, 0.5, 1, 2, 4, 8], min_length=1
    )
    inhibition_factors: list[float] = Field(default_factory=lambda: [1], min_length=1)
    warmup_ms: float = Field(default=500, gt=0)
    pulse_ms: float = Field(default=20, gt=0)
    post_ms: float = Field(default=2000, gt=0)
    stimulus_fraction: float = Field(default=0.05, ge=0, le=1)
    stimulus_amplitude: float = Field(default=0.4, ge=0)
    rate_bin_ms: float = Field(default=5, gt=0)
    tail_ms: float = Field(default=100, gt=0)
    sample_ms: float = Field(default=1, gt=0)

    @model_validator(mode="after")
    def check_consistency(self):
        if self.n_exc >= self.n:
            raise ValueError("E/Iの両集団に少なくとも1細胞必要です")
        for values in (self.wiring_seeds, self.stimulus_seeds):
            if min(values) < 0 or len(set(values)) != len(values):
                raise ValueError("シードは非負かつ重複なしが必要です")
        for values in (self.multipliers, self.inhibition_factors):
            if (
                not np.all(np.isfinite(values))
                or min(values) < 0
                or len(set(values)) != len(values)
            ):
                raise ValueError("走査値は非負・有限・重複なしが必要です")
        if 0 not in self.multipliers:
            raise ValueError("走査には結合倍率0の対照が必要です")
        if self.tail_ms > self.post_ms:
            raise ValueError("末尾判定窓が観測区間を超えています")
        arrival_to_dendritic_ms([self.arrival_ms], self.dt_ms)
        for name in (
            "warmup_ms",
            "pulse_ms",
            "post_ms",
            "rate_bin_ms",
            "tail_ms",
            "sample_ms",
        ):
            steps_for(getattr(self, name), self.dt_ms, name)
        for interval in (self.warmup_ms, self.pulse_ms, self.post_ms):
            steps_for(interval, self.sample_ms, "sample interval")
            steps_for(interval, self.rate_bin_ms, "rate interval")
        return self

    @property
    def end_ms(self):
        return self.warmup_ms + self.pulse_ms + self.post_ms


def dump_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def dump_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_snapshot(cfg, wiring_seed, path):
    rng = np.random.default_rng(wiring_seed)
    row_blocks, col_blocks = [], []
    # 行ごとに生成して一時N×N行列を避ける。配置乱数とは独立。
    for row in range(cfg.n):
        cols = np.flatnonzero(rng.random(cfg.n) < cfg.probability)
        cols = cols[cols != row]
        row_blocks.append(np.full(cols.size, row, dtype=np.int32))
        col_blocks.append(cols.astype(np.int32))
    rows, cols = np.concatenate(row_blocks), np.concatenate(col_blocks)
    coords = np.empty((cfg.n, 0), dtype=np.float64)
    if cfg.space.kind == "uniform_2d":
        coords = np.random.default_rng(cfg.space.seed).uniform(
            0, cfg.space.extent, (cfg.n, 2)
        )
    orders = np.array(
        [
            np.random.default_rng(
                np.random.SeedSequence([cfg.selection_seed, seed, 0])
            ).permutation(cfg.n_exc)
            for seed in cfg.stimulus_seeds
        ],
        dtype=np.int32,
    )
    arrays = dict(
        ids=np.arange(cfg.n, dtype=np.int32),
        polarity=np.r_[np.zeros(cfg.n_exc), np.ones(cfg.n - cfg.n_exc)].astype(np.int8),
        rows=rows,
        cols=cols,
        arrival_ms=np.full(rows.size, cfg.arrival_ms),
        coords=coords,
        stimulus_orders=orders,
        autapse_exc_order=np.random.default_rng(
            np.random.SeedSequence([cfg.selection_seed, 1])
        )
        .permutation(cfg.n_exc)
        .astype(np.int32),
        autapse_inh_order=(
            cfg.n_exc
            + np.random.default_rng(
                np.random.SeedSequence([cfg.selection_seed, 2])
            ).permutation(cfg.n - cfg.n_exc)
        ).astype(np.int32),
    )
    meta = dict(
        version=1,
        n=cfg.n,
        n_exc=cfg.n_exc,
        wiring_seed=wiring_seed,
        probability=cfg.probability,
        connection_rule=cfg.connection_rule,
        delay_rule=cfg.delay_rule,
        arrival_ms=cfg.arrival_ms,
        stimulus_seeds=cfg.stimulus_seeds,
        selection_seed=cfg.selection_seed,
        space=cfg.space.model_dump(),
    )
    return save_snapshot(path, arrays, meta)


def make_app_config(cfg, snapshot_path, multiplier, inhibition_factor):
    from src.core.config_manager import AppConfig

    arrays, meta, fingerprint = load_snapshot(snapshot_path)
    if (meta["n"], meta["n_exc"]) != (cfg.n, cfg.n_exc):
        raise ValueError("配線のN/E/Iと実験設定が一致しません")
    if meta["arrival_ms"] != cfg.arrival_ms or meta["probability"] != cfg.probability:
        raise ValueError("配線の遅延・接続確率と実験設定が一致しません")
    if meta["space"] != cfg.space.model_dump():
        raise ValueError("配線の空間条件と実験設定が一致しません")
    if (
        not np.isfinite(multiplier)
        or not np.isfinite(inhibition_factor)
        or min(multiplier, inhibition_factor) < 0
    ):
        raise ValueError("重み倍率は有限・非負が必要です")
    profiles = yaml.safe_load((ROOT / "configs/components/neurons.yaml").read_text())
    neurons = {}
    for label, cell, count, polarity in (
        ("Exc", cfg.exc_model, cfg.n_exc, "excitatory"),
        ("Inh", cfg.inh_model, cfg.n - cfg.n_exc, "inhibitory"),
    ):
        neurons[label] = dict(
            type=cell.type,
            mode=cell.mode,
            num=count,
            polarity=polarity,
            **profiles[cell.type][cell.mode],
        )
        if "Ioffset" in neurons[label]:
            neurons[label]["Ioffset"] = 0.0
    weight = {
        key: value * multiplier * (inhibition_factor if key[0] == "i" else 1)
        for key, value in cfg.weights.model_dump().items()
    }
    saved = dict(path=str(Path(snapshot_path).resolve()), fingerprint=fingerprint)
    task_duration = cfg.end_ms
    return AppConfig.model_validate(
        dict(
            simulation=dict(
                N=cfg.n, dt=cfg.dt_ms, seed=meta["wiring_seed"], backend=cfg.backend
            ),
            inputs=dict(GaussianNoise=dict(enable=False)),
            neurons=neurons,
            synapses={
                label: dict(
                    source=[label],
                    synapse=dict(type="ExpCurr", tau=tau),
                    plasticity=dict(type="StaticPulseDendriticDelay"),
                )
                for label, tau in (("Exc", cfg.tau_exc_ms), ("Inh", cfg.tau_inh_ms))
            },
            network=dict(
                space=dict(profile_name="saved_snapshot", **saved),
                connection=dict(profile_name="saved_snapshot", **saved),
                weight=dict(profile_name="baseline_pair_weights", **weight),
                delay=dict(profile_name="saved_arrival_delay", dt=cfg.dt_ms, **saved),
                sparse="force",
            ),
            layout=dict(assignment="sequential"),
            task=dict(
                profile_name="baseline_network",
                duration=cfg.end_ms,
                record_buffer_ms=task_duration,
            ),
            meta=dict(timestamp="baseline_network"),
        )
    )


def summarize_spikes(times, ids, cfg):
    times, ids = np.asarray(times), np.asarray(ids)
    stop = cfg.warmup_ms + cfg.pulse_ms
    edges = np.arange(steps_for(cfg.end_ms, cfg.rate_bin_ms) + 1) * cfg.rate_bin_ms
    exc, _ = np.histogram(times[ids < cfg.n_exc], edges)
    inh, _ = np.histogram(times[ids >= cfg.n_exc], edges)
    rates = np.column_stack(
        (
            edges[:-1],
            exc * 1000 / (cfg.n_exc * cfg.rate_bin_ms),
            inh * 1000 / ((cfg.n - cfg.n_exc) * cfg.rate_bin_ms),
        )
    )
    post_mask = (times >= stop) & (times < cfg.end_ms)
    post = times[post_mask]
    tail = (times >= cfg.end_ms - cfg.tail_ms) & (times < cfg.end_ms)
    counts = (exc + inh)[edges[:-1] >= stop]
    peak_fraction = 0.0
    if post.size:
        bin_id = np.floor((post - stop) / cfg.rate_bin_ms).astype(np.int64)
        pairs = np.unique(bin_id * cfg.n + ids[post_mask])
        peak_fraction = float(np.bincount(pairs // cfg.n).max() / cfg.n)
    spectrum = (
        np.abs(np.fft.rfft((counts - counts.mean()) * np.hanning(counts.size))) ** 2
    )
    spectrum[0] = 0
    peak_hz, peak_power = None, None
    if spectrum.sum() > 0 and post.size >= 5:
        index = int(spectrum.argmax())
        peak_hz = float(np.fft.rfftfreq(counts.size, cfg.rate_bin_ms / 1000)[index])
        peak_power = float(spectrum[index] / spectrum.sum())
    during = (times >= cfg.warmup_ms) & (times < stop)
    metrics = dict(
        pre_spikes=int(np.count_nonzero(times < cfg.warmup_ms)),
        during_spikes=int(during.sum()),
        during_participants=int(np.unique(ids[during]).size),
        post_spikes=int(post.size),
        post_participants=int(np.unique(ids[post_mask]).size),
        post_exc_spikes=int(np.count_nonzero(post_mask & (ids < cfg.n_exc))),
        post_inh_spikes=int(np.count_nonzero(post_mask & (ids >= cfg.n_exc))),
        post_rate_hz=float(post.size * 1000 / (cfg.n * cfg.post_ms)),
        last_post_latency_ms=float(post[-1] - stop) if post.size else None,
        tail_spikes=int(tail.sum()),
        tail_participants=int(np.unique(ids[tail]).size),
        peak_active_fraction_bin=peak_fraction,
        population_count_cv=(
            float(counts.std() / counts.mean()) if counts.mean() > 0 else None
        ),
        spectrum_peak_hz=peak_hz,
        spectrum_peak_fraction=peak_power,
        activity_status=(
            "active_at_observation_tail"
            if tail.any()
            else "post_activity_without_tail" if post.size else "no_post_activity"
        ),
    )
    return metrics, rates


def _case_worker(cfg, snapshot_path, out, build_dir, kwargs):
    """ネイティブモデルの寿命と共有ライブラリを試行ごとに分離する。"""
    with Path(out).with_suffix(".log").open("w") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)

    sys.stdout.reconfigure(line_buffering=True)
    faulthandler.enable()

    try:
        row = _run_case(cfg, snapshot_path, out, build_dir, **kwargs)
        dump_json(Path(out) / "summary.json", row)
    except Exception as exc:
        Path(out).mkdir(parents=True, exist_ok=True)
        dump_json(
            Path(out) / "worker_error.json",
            {
                "type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def run_case(cfg, snapshot_path, out, build_dir, **kwargs):
    """各試行を独立プロセスで実行し、GeNNのネイティブ状態を共有しない。"""
    if Path(out).exists():
        raise FileExistsError(out)

    out = Path(out)
    attempts = []
    for attempt in range(3):
        trial_build = (
            Path(build_dir)
            if attempt == 0
            else Path(build_dir).with_name(f"{Path(build_dir).name}_retry_{attempt}")
        )
        process = multiprocessing.get_context("spawn").Process(
            target=_case_worker,
            args=(cfg, snapshot_path, out, trial_build, kwargs),
        )
        process.start()
        process.join()
        if process.exitcode == 0:
            dump_json(out / "build_retries.json", attempts)
            return json.loads((out / "summary.json").read_text())

        error_path = out / "worker_error.json"
        if error_path.exists():
            dump_json(out / "build_retries.json", attempts)
            error = json.loads(error_path.read_text())
            if error["type"] == "FloatingPointError":
                raise FloatingPointError(error["error"])
            raise RuntimeError(error["error"])

        out.mkdir(parents=True, exist_ok=True)
        status_path = out / "status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        message = f"実行プロセスが異常終了しました: exitcode={process.exitcode}"
        dump_json(status_path, {**previous, "status": "failed", "error": message})

        # 観測結果で試行を選別しない。シミュレーション開始前のSIGSEGVだけ再試行する。
        if (
            process.exitcode != -11
            or previous.get("phase") != "before_simulation"
            or attempt == 2
        ):
            dump_json(out / "build_retries.json", attempts)
            raise RuntimeError(message)

        archive = out.with_name(f"{out.name}_build_failure_{attempt + 1}")
        out.rename(archive)
        log = out.with_suffix(".log")
        if log.exists():
            log.rename(archive.with_suffix(".log"))
        attempts.append({"exitcode": process.exitcode, "archive": str(archive)})
        print(
            f"{out.name}: GeNNコード生成の異常終了を保存し再試行 ({attempt + 1}/2)",
            flush=True,
        )


def _run_case(
    cfg,
    snapshot_path,
    out,
    build_dir,
    multiplier=1.0,
    inhibition_factor=1.0,
    stimulus_seed=None,
    no_stimulus=False,
    amplitude_vector=None,
):
    """毎回新しいモデルをロードする。状態・遅延キューを試行間で共有しない。"""
    import pygenn
    from src.core.NetworkBuilder import NetworkBuilder
    from src.core.simulator import GeNNSimulator
    import src.models.network.snapshot  # noqa: F401
    import src.models.neurons.pqn_float  # noqa: F401
    import src.models.plasticity.standard_models  # noqa: F401
    import src.models.synapses.standard_models  # noqa: F401

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    arrays, meta, fingerprint = load_snapshot(snapshot_path)
    stimulus_seed = (
        meta["stimulus_seeds"][0] if stimulus_seed is None else stimulus_seed
    )
    index = meta["stimulus_seeds"].index(stimulus_seed)
    count = int(np.floor(cfg.n_exc * cfg.stimulus_fraction + 0.5))
    targets = arrays["stimulus_orders"][index, :count]
    amplitudes = np.zeros(cfg.n)
    if not no_stimulus:
        amplitudes[targets] = cfg.stimulus_amplitude
    if amplitude_vector is not None:
        amplitudes = np.asarray(amplitude_vector, dtype=float)
        if amplitudes.shape != (cfg.n,) or not np.all(np.isfinite(amplitudes)):
            raise ValueError("校正入力ベクトルが不正です")
        targets = np.flatnonzero(amplitudes)

    # 設定・入力・出典を実行前に保存し、失敗した条件も追跡できるようにする。
    app = make_app_config(cfg, snapshot_path, multiplier, inhibition_factor)
    (out / "settings.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(), allow_unicode=True, sort_keys=False)
    )
    (out / "resolved.yaml").write_text(
        yaml.safe_dump(app.model_dump(), allow_unicode=True, sort_keys=False)
    )
    np.save(out / "stimulus_ids.npy", targets)
    np.save(out / "stimulus_amplitudes.npy", amplitudes)
    source_files = [
        Path(__file__),
        ROOT / "scripts/baseline_network.py",
        ROOT / "src/core/network_snapshot.py",
        ROOT / "src/models/network/snapshot.py",
        ROOT / "src/core/NetworkBuilder.py",
        ROOT / "src/core/simulator.py",
        ROOT / "src/models/neurons/pqn_float.py",
        ROOT / "src/models/neurons/pqn_origin.py",
    ]
    info = dict(
        snapshot_fingerprint=fingerprint,
        connectivity_hash=array_hash(
            arrays, ("ids", "polarity", "rows", "cols", "arrival_ms")
        ),
        snapshot_path=str(Path(snapshot_path).resolve()),
        precision="double",
        backend=cfg.backend,
        dt_ms=cfg.dt_ms,
        wiring_seed=meta["wiring_seed"],
        stimulus_seed=stimulus_seed,
        multiplier=multiplier,
        inhibition_factor=inhibition_factor,
        stimulus_count=int(targets.size),
        stimulus_exc_count=int(np.count_nonzero(amplitudes[: cfg.n_exc])),
        stimulus_inh_count=int(np.count_nonzero(amplitudes[cfg.n_exc :])),
        stimulus_exc_fraction=float(
            np.count_nonzero(amplitudes[: cfg.n_exc]) / cfg.n_exc
        ),
        stimulus_inh_fraction=float(
            np.count_nonzero(amplitudes[cfg.n_exc :]) / (cfg.n - cfg.n_exc)
        ),
        no_stimulus=no_stimulus,
        pulse_start_ms=cfg.warmup_ms,
        pulse_end_ms=cfg.warmup_ms + cfg.pulse_ms,
        observation_end_ms=cfg.end_ms,
        input_unit="PQN model units",
        voltage_unit="PQN model units",
        time_convention="スパイク・入力は更新開始時刻、状態は更新終了時刻。区間は左閉右開。",
        fresh_state=True,
        within_population_homogeneous=True,
        all_neurons_recorded=True,
        autapse_exc_count=0,
        autapse_inh_count=0,
        absolute_weights={
            key: float(np.float32(getattr(app.network.weight, key)))
            for key in ("ee", "ei", "ie", "ii")
        },
        source_hashes={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
    )
    dump_json(out / "metadata.json", info)
    dump_json(out / "status.json", {"status": "running", "phase": "before_simulation"})
    builder = NetworkBuilder(
        app, model_name="BaselineNetwork", code_gen_dir=str(Path(build_dir).resolve())
    )
    loaded = False
    try:
        model, layout = builder.build(rec_spike=True)
        (out / "resolved.yaml").write_text(
            yaml.safe_dump(app.model_dump(), allow_unicode=True, sort_keys=False)
        )
        actual_pairs = (
            np.concatenate(
                [
                    np.column_stack((i.global_src, i.global_tgt))
                    for i in builder.synapse_index.values()
                ]
            )
            if builder.synapse_index
            else np.empty((0, 2), int)
        )
        codes = np.sort(actual_pairs[:, 0] * cfg.n + actual_pairs[:, 1])
        if not np.array_equal(codes, arrays["rows"] * cfg.n + arrays["cols"]):
            raise RuntimeError("構築した接続が保存COOと一致しません")

        # 浮動小数点の時刻加算誤差を避け、ステップ中央で半開区間を判定する。
        stimulus_model = pygenn.create_current_source_model(
            "BaselinePulse",
            params=["Start", "Stop"],
            vars=[("Amplitude", "scalar"), ("Injected", "scalar")],
            injection_code="""
            Injected = (t + dt * 0.5 >= Start && t + dt * 0.5 < Stop) ? Amplitude : 0.0;
            injectCurrent(Injected);
            """,
        )
        currents = {}
        neuron_constants = {}
        from src.core.registry import NEURON_MODELS

        for label, pop in model.neuron_populations.items():
            ids = layout.global_indices(label)
            currents[label] = model.add_current_source(
                label + "Pulse",
                stimulus_model,
                pop,
                {"Start": cfg.warmup_ms, "Stop": cfg.warmup_ms + cfg.pulse_ms},
                {"Amplitude": amplitudes[ids], "Injected": 0.0},
            )
            neuron = NEURON_MODELS.get(app.neurons[label].type)(
                app.neurons[label], cfg.dt_ms
            )
            neuron_constants[label] = {
                "params": neuron.params,
                "initial_vars": neuron.vars,
            }
        dump_json(out / "neuron_constants.json", neuron_constants)
        sim = GeNNSimulator(model, app, builder)
        step_count = steps_for(cfg.end_ms, cfg.dt_ms)
        # GeNNは満杯の記録窓のみ取得できる。整数ステップ数を明示する。
        sim.max_timesteps = step_count
        sim.setup(backup_initial_states=False)
        loaded = True
        dump_json(out / "status.json", {"status": "running", "phase": "simulation"})
        sample_steps = steps_for(cfg.sample_ms, cfg.dt_ms)
        representatives = np.unique(
            np.r_[
                targets[:3],
                np.arange(min(3, cfg.n_exc)),
                cfg.n_exc + np.arange(min(3, cfg.n - cfg.n_exc)),
            ]
        )
        if amplitude_vector is not None:
            # 独立細胞の校正では全条件の波形を保存し、発火検出も確認する。
            representatives = np.arange(cfg.n)

        state_keys = ("V", "N", "Q")
        traces = {key: [] for key in state_keys}
        recorded_times, input_trace = [], []
        warmup_reference = None
        drift = None
        for first_step in range(0, step_count, sample_steps):
            sim.step(min(sample_steps, step_count - first_step))
            t = min(first_step + sample_steps, step_count) * cfg.dt_ms
            full_state = {}
            for key in state_keys:
                # 既存sim.pullはfloat32へ結合するため、実行時のdouble精度を保持する。
                local_values = {}
                for label, pop in model.neuron_populations.items():
                    pop.vars[key].pull_from_device()
                    local_values[label] = pop.vars[key].view.copy()
                values = layout.merge_local_to_global(local_values, dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    np.savez_compressed(
                        out / "failed_state.npz", time_ms=t, variable=key, values=values
                    )
                    raise FloatingPointError(f"{t} msで{key}に非有限値")
                full_state[key] = values
                traces[key].append(values[representatives])
            recorded_times.append(t)
            max_input = []
            for current in currents.values():
                current.vars["Injected"].pull_from_device()
                max_input.append(float(np.max(np.abs(current.vars["Injected"].view))))
            input_trace.append(max_input)
            if np.isclose(t, max(cfg.sample_ms, cfg.warmup_ms - 50)):
                warmup_reference = {k: v.copy() for k, v in full_state.items()}
            if np.isclose(t, cfg.warmup_ms) and warmup_reference is not None:
                drift = {
                    k: float(np.max(np.abs(full_state[k] - warmup_reference[k])))
                    for k in state_keys
                }

        # 全観測窓のスパイクを保存し、図と集計を同じ生データから生成する。
        spikes = sim.get_global_spikes()
        times = np.rint(spikes["times"] / cfg.dt_ms) * cfg.dt_ms
        ids = spikes["ids"]
        if times.size and (times.min() < 0 or times.max() >= cfg.end_ms):
            raise RuntimeError("観測区間外のスパイク時刻")
        np.savez_compressed(out / "spikes.npz", times_ms=times, ids=ids)
        np.savez_compressed(
            out / "traces.npz",
            time_end_ms=np.array(recorded_times),
            input_time_start_ms=np.array(recorded_times) - cfg.dt_ms,
            ids=representatives,
            max_abs_input=np.array(input_trace),
            **{k: np.array(v) for k, v in traces.items()},
        )
        metrics, rates = summarize_spikes(times, ids, cfg)
        stimulated_during = (
            (times >= cfg.warmup_ms)
            & (times < cfg.warmup_ms + cfg.pulse_ms)
            & np.isin(ids, targets)
        )
        metrics["stimulated_during_spikes"] = int(stimulated_during.sum())
        metrics["stimulated_during_participants"] = int(
            np.unique(ids[stimulated_during]).size
        )
        dump_json(out / "warmup_drift.json", drift)
        dump_json(out / "metrics.json", metrics)
        np.savetxt(
            out / "rates.csv",
            rates,
            delimiter=",",
            header="bin_start_ms,exc_hz,inh_hz",
            comments="",
        )
        plot_run(out, times, ids, rates, cfg)
        dump_json(out / "status.json", {"status": "complete"})
        return {
            **{
                k: info[k]
                for k in (
                    "wiring_seed",
                    "stimulus_seed",
                    "multiplier",
                    "inhibition_factor",
                    "no_stimulus",
                )
            },
            "run_dir": str(out),
            **metrics,
        }
    except Exception as exc:
        dump_json(out / "status.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        if loaded:
            model.unload()


def plot_run(out, times, ids, rates, cfg):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # 生スパイクは全件保存。図の描画だけを間引く。
    stride = max(1, times.size // 100000)
    axes[0].scatter(times[::stride], ids[::stride], s=0.4, rasterized=True)
    axes[0].axhline(cfg.n_exc - 0.5, color="gray", lw=0.5)
    axes[0].set_ylabel("Neuron ID")
    axes[1].plot(rates[:, 0], rates[:, 1], label="Exc")
    axes[1].plot(rates[:, 0], rates[:, 2], label="Inh")
    axes[1].set(xlabel="Time (ms)", ylabel="Population rate (Hz)")
    axes[1].legend()
    for ax in axes:
        ax.axvspan(
            cfg.warmup_ms, cfg.warmup_ms + cfg.pulse_ms, color="orange", alpha=0.2
        )
    fig.tight_layout()
    fig.savefig(out / "activity.png", dpi=140)
    plt.close(fig)
