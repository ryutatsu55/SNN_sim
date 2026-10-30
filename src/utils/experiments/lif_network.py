"""確率LIF・STDP・短期抑圧による基準ネットワークの発達と固定後観測。"""

import csv
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

ROOT = Path(__file__).resolve().parents[3]


def save_json(path, value):
    """監視側が書きかけのJSONを読まないよう置換する。"""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(directory, exclude=()):
    directory = Path(directory)
    return {str(p.relative_to(directory)): dict(size=p.stat().st_size, sha256=sha256(p))
            for p in sorted(directory.rglob("*")) if p.is_file()
            and not any(p.relative_to(directory).parts[0] == item for item in exclude)}


class NetworkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    n: int = Field(default=1000, ge=2)
    n_exc: int = Field(default=800, ge=1)
    probability: float = Field(default=0.1, gt=0, le=1)
    seeds: list[int] = Field(default_factory=lambda: [11, 22, 33], min_length=1)
    dt_ms: float = Field(default=0.1, gt=0)
    backend: Literal["cpu", "cuda"] = "cpu"
    development_ms: float = Field(default=3600000, gt=0)
    frozen_ms: float = Field(default=60000, gt=0)
    chunk_ms: float = Field(default=10000, gt=0)
    snapshot_ms: float = Field(default=60000, gt=0)
    gmax_values: list[float] = Field(default_factory=lambda: [1.5, 2.5, 4], min_length=1)
    beta_inh_values: list[float] = Field(default_factory=lambda: [1.05, 1.15, 1.25], min_length=1)
    rest_rates_hz: list[float] = Field(default_factory=lambda: [0.4], min_length=1)
    beta_exc: float = Field(default=1.05, gt=0)
    learning_amplitude: float = Field(default=0.01, gt=0)
    escape_gain_mv: float = Field(default=4, gt=0)
    max_rate_hz: float = Field(default=100, gt=0)
    case_timeout_s: float = Field(default=86400, gt=0)

    @model_validator(mode="after")
    def consistent(self):
        if self.n_exc >= self.n:
            raise ValueError("興奮性・抑制性の両集団が必要です")
        for values in (self.seeds, self.gmax_values, self.beta_inh_values, self.rest_rates_hz):
            if len(set(values)) != len(values) or not np.all(np.isfinite(values)) or min(values) <= 0:
                raise ValueError("シード・走査値は正・有限・重複なしが必要です")
        if max(self.beta_inh_values) >= 2:
            raise ValueError("抑制STDPには0 < beta_I < 2が必要です")
        if max(self.rest_rates_hz) * self.dt_ms / 1000 >= 1:
            raise ValueError("静止時の1ステップ発火確率が1以上です")
        for duration in (self.chunk_ms, 0.8, 1.5):
            if not np.isclose(duration / self.dt_ms, round(duration / self.dt_ms), atol=1e-8, rtol=0):
                raise ValueError("記録間隔と遅延はdtの整数倍にしてください")
        for duration in (self.development_ms, self.frozen_ms, self.snapshot_ms):
            if not np.isclose(duration / self.chunk_ms, round(duration / self.chunk_ms), atol=1e-8, rtol=0):
                raise ValueError("発達・固定観測・重み保存間隔はchunk_msの整数倍にしてください")
        return self


def load_settings(path):
    return NetworkSettings.model_validate(yaml.safe_load(Path(path).read_text()))


def preset_settings(cfg, preset):
    if preset == "full":
        return cfg
    # 動作確認も1000細胞を維持し、時間と条件数だけを絞る。
    durations = dict(smoke=(1000, 1000, 500), pilot=(60000, 10000, 1000))
    develop, frozen, chunk = durations[preset]
    return NetworkSettings.model_validate({**cfg.model_dump(), "seeds": cfg.seeds[:1],
               "gmax_values": [cfg.gmax_values[len(cfg.gmax_values) // 2]],
               "beta_inh_values": [cfg.beta_inh_values[len(cfg.beta_inh_values) // 2]],
               "rest_rates_hz": cfg.rest_rates_hz[:1], "development_ms": develop,
               "frozen_ms": frozen, "chunk_ms": chunk, "snapshot_ms": chunk})


def make_plan(cfg):
    return [dict(case_id=f"case_{i:04d}", seed=seed, gmax=g, beta_inh=beta, rest_rate_hz=rate)
            for i, (g, beta, rate, seed) in enumerate(itertools.product(
                cfg.gmax_values, cfg.beta_inh_values, cfg.rest_rates_hz, cfg.seeds))]


def app_config(cfg, case):
    """共有プロファイルを暗黙に使わず、実際の定数を全て保存できる設定を作る。"""
    from src.core.config_manager import AppConfig
    neurons, synapses = {}, {}
    for name, count, polarity, refractory in (
        ("Exc", cfg.n_exc, "excitatory", 3), ("Inh", cfg.n - cfg.n_exc, "inhibitory", 2),
    ):
        excitatory = name == "Exc"
        neurons[name] = dict(type="akita_escape_lif", mode=polarity, num=count,
                             polarity=polarity, tau_m=30, v_rest=-74, v_th=-54,
                             b=cfg.escape_gain_mv, f_rest=case["rest_rate_hz"], tau_refrac=refractory)
        plasticity = dict(type="custom_Akita", mode="e-stdp" if excitatory else "i-stdp",
                          pairing="nearest", tau_rec=150, U=0.4, g_max=case["gmax"],
                          normalize_gmax_by_fan_in=False, Wmin=0, Wmax=1)
        if excitatory:
            plasticity.update(A_E=cfg.learning_amplitude, tau_E=20, beta_E=cfg.beta_exc)
        else:
            plasticity.update(A_I=cfg.learning_amplitude, tau_I1=10, tau_I2=20, beta_I=case["beta_inh"])
        synapses[name] = dict(source=[name], plasticity=plasticity,
                              synapse=dict(type="ExpCond", tau=2 if excitatory else 4, E=0 if excitatory else -80))
    return AppConfig.model_validate(dict(
        simulation=dict(N=cfg.n, dt=cfg.dt_ms, seed=case["seed"], backend=cfg.backend),
        inputs=dict(GaussianNoise=dict(enable=False)), neurons=neurons, synapses=synapses,
        network=dict(area=dict(profile_name="no_space"), space=dict(profile_name="no_space"),
                     connection=dict(profile_name="constant_prob", p=cfg.probability, allow_self_connections=False),
                     weight=dict(profile_name="constant_zero", base_weight=0),
                     delay=dict(profile_name="type_based_axonal", d_ee=1.5, d_ei=0.8, d_ie=0.8, d_ii=0.8)),
        layout=dict(assignment="sequential"), task=dict(profile_name="lif_network", record_buffer_ms=cfg.chunk_ms),
        meta=dict(timestamp="lif_network")))


def pull_weights(model, builder):
    """保存用の値をCOO順に散布し、doubleを維持する。"""
    coo = builder.global_coo()
    result = np.empty(coo.num_synapses, dtype=np.float64)
    for name, index in builder.synapse_index.items():
        variable = model.synapse_populations[name].vars["w"]
        variable.pull_from_device()
        result[index.global_positions] = np.asarray(variable.values, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < -1e-12) or np.any(result > 1 + 1e-12):
        raise FloatingPointError("重みが非有限値または0〜1の範囲外です")
    return result


def freeze_learning(model):
    """STP・電位・乱数・遅延列を維持し、STDPだけを停止する。"""
    for name, group in model.synapse_populations.items():
        group.set_dynamic_param_value("A_E" if name.startswith("Exc_to_") else "C_I", 0.0)


def collect_state(model, layout):
    state = {}
    for key in ("V", "RefracTime", "Isyn_rec"):
        values = {}
        for label, pop in model.neuron_populations.items():
            pop.vars[key].pull_from_device()
            values[label] = pop.vars[key].view.copy()
            if not np.all(np.isfinite(values[label])):
                raise FloatingPointError(f"{label}/{key}が非有限値です")
        state[key] = layout.merge_local_to_global(values, dtype=np.float64)
    return state


def save_boundary_state(path, model, layout):
    """STP資源等を記録する。乱数・遅延キューを含む完全チェックポイントではない。"""
    state = collect_state(model, layout)
    for group_name, group in model.synapse_populations.items():
        for key, variable in group.pre_vars.items():
            variable.pull_from_device()
            state[f"{group_name}__{key}"] = variable.view.copy()
    np.savez_compressed(path, time_ms=float(model.t), **state)


def run_case(cfg, case, out, build_dir):
    import src.models.neurons.akita_escape_lif  # noqa: F401
    import src.models.network.connectors  # noqa: F401
    import src.models.network.delays  # noqa: F401
    import src.models.network.space  # noqa: F401
    import src.models.network.weights  # noqa: F401
    import src.models.plasticity.custom_Akita  # noqa: F401
    import src.models.synapses.standard_models  # noqa: F401
    from src.core.NetworkBuilder import NetworkBuilder
    from src.core.simulator import GeNNSimulator

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "chunks").mkdir()
    (out / "weights").mkdir()
    app = app_config(cfg, case)
    (out / "resolved.yaml").write_text(yaml.safe_dump(app.model_dump(), allow_unicode=True))
    save_json(out / "case.json", case)
    save_json(out / "settings.json", cfg.model_dump())
    started = time.monotonic()
    save_json(out / "progress.json", dict(status="building", phase="before_simulation"))
    builder = NetworkBuilder(app, model_name="LIFPlasticNetwork", code_gen_dir=str(build_dir))
    model, layout = builder.build(rec_spike=True)
    for name, group in model.synapse_populations.items():
        group.set_wu_param_dynamic("A_E" if name.startswith("Exc_to_") else "C_I")
    coo = builder.global_coo()
    if np.any(coo.row == coo.col):
        raise ValueError("基準ネットワークに自己結合があります")
    np.savez_compressed(out / "connectivity.npz", row=coo.row, col=coo.col, delays_ms=coo.delays,
                        n=cfg.n, n_exc=cfg.n_exc)
    layout.save_axes(out / "layout_axes.npz")
    from src.core.registry import NEURON_MODELS
    constants = {}
    for name, neuron_config in app.neurons.items():
        neuron = NEURON_MODELS.get(neuron_config.type)(neuron_config, cfg.dt_ms)
        constants[name] = dict(params=neuron.params, initial_vars=neuron.vars)
    save_json(out / "neuron_constants.json", constants)
    save_json(out / "metadata.json", dict(precision="double", conductance_unit="leak_conductance_ratio",
              voltage_unit="mV", time_unit="ms", wiring_seed=case["seed"], noise_seed=case["seed"],
              synapses=int(coo.num_synapses), connectivity_sha256=sha256(out / "connectivity.npz"),
              stdp_pairing="nearest", stdp_time="presynaptic_arrival", autapses=0,
              state_samples="チャンク末尾の代表細胞。毎ステップの波形ではない。"))
    sim = GeNNSimulator(model, app, builder)
    try:
        sim.setup(backup_initial_states=False)
        setup_s = time.monotonic() - started
        np.savez_compressed(out / "weights/initial.npz", w=pull_weights(model, builder), time_ms=0)
        freeze_at = int(round(cfg.development_ms / cfg.chunk_ms))
        count = freeze_at + int(round(cfg.frozen_ms / cfg.chunk_ms))
        snapshot_every = int(round(cfg.snapshot_ms / cfg.chunk_ms))
        frozen_weights = None
        representatives = np.unique(np.r_[np.arange(min(4, cfg.n_exc)), cfg.n_exc + np.arange(min(4, cfg.n - cfg.n_exc))])
        masks = [(coo.row < cfg.n_exc) & (coo.col < cfg.n_exc),
                 (coo.row < cfg.n_exc) & (coo.col >= cfg.n_exc),
                 (coo.row >= cfg.n_exc) & (coo.col < cfg.n_exc),
                 (coo.row >= cfg.n_exc) & (coo.col >= cfg.n_exc)]
        header = ["chunk", "start_ms", "end_ms", "phase", "spikes", "rate_exc_hz", "rate_inh_hz",
                  "participants", "mean_ee", "mean_ei", "mean_ie", "mean_ii", "weight_at_max_fraction"]
        with (out / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            for chunk in range(count):
                if chunk == freeze_at:
                    frozen_weights = pull_weights(model, builder)
                    np.savez_compressed(out / "weights/frozen.npz", w=frozen_weights, time_ms=cfg.development_ms)
                    # 停止前の細胞・STP状態は対照と後続実験用の記録。完全再開用ではない。
                    save_boundary_state(out / "freeze_state.npz", model, layout)
                    freeze_learning(model)
                start_ms = chunk * cfg.chunk_ms
                sim.step(sim.max_timesteps)
                spikes = sim.get_global_spikes()
                local = np.asarray(spikes["times"], dtype=np.float64) - start_ms
                ids = np.asarray(spikes["ids"], dtype=np.int32)
                if not np.all(np.isfinite(local)) or (local.size and (local.min() < -cfg.dt_ms * 0.01 or local.max() >= cfg.chunk_ms)):
                    raise ValueError("スパイク記録の時刻がチャンク範囲外です")
                phase = "development" if chunk < freeze_at else "frozen"
                np.savez_compressed(out / "chunks" / f"{chunk:06d}.npz", times_ms=local, ids=ids,
                                    start_ms=start_ms, duration_ms=cfg.chunk_ms, phase=phase)
                weight = pull_weights(model, builder)
                if frozen_weights is not None and not np.array_equal(weight, frozen_weights):
                    raise RuntimeError("STDP停止後に重みが変化しました")
                if (chunk + 1) % snapshot_every == 0 or chunk + 1 == count:
                    np.savez_compressed(out / "weights" / f"{chunk + 1:06d}.npz", w=weight, time_ms=start_ms + cfg.chunk_ms)
                state = collect_state(model, layout)
                np.savez_compressed(out / "chunks" / f"state_{chunk:06d}.npz", ids=representatives,
                                    time_ms=start_ms + cfg.chunk_ms, **{key: value[representatives] for key, value in state.items()})
                row = dict(chunk=chunk, start_ms=start_ms, end_ms=start_ms + cfg.chunk_ms, phase=phase,
                           spikes=int(ids.size), rate_exc_hz=float(np.count_nonzero(ids < cfg.n_exc) * 1000 / cfg.n_exc / cfg.chunk_ms),
                           rate_inh_hz=float(np.count_nonzero(ids >= cfg.n_exc) * 1000 / (cfg.n - cfg.n_exc) / cfg.chunk_ms),
                           participants=int(np.unique(ids).size), weight_at_max_fraction=float(np.mean(weight >= 1 - 1e-9)) if weight.size else 0)
                row.update({key: float(weight[mask].mean()) if mask.any() else None
                            for key, mask in zip(("mean_ee", "mean_ei", "mean_ie", "mean_ii"), masks)})
                writer.writerow(row)
                stream.flush()
                save_json(out / "progress.json", dict(status="running", completed_chunks=chunk + 1,
                          total_chunks=count, phase=phase, simulation_ms=start_ms + cfg.chunk_ms,
                          elapsed_s=time.monotonic() - started))
                print(f"{case['case_id']} {chunk + 1}/{count} {phase}: E={row['rate_exc_hz']:.3f}, I={row['rate_inh_hz']:.3f} Hz", flush=True)
                if max(row["rate_exc_hz"], row["rate_inh_hz"]) > cfg.max_rate_hz:
                    raise RuntimeError(f"発火率が上限{cfg.max_rate_hz}Hzを超えました。候補から除外します")
        save_boundary_state(out / "final_state.npz", model, layout)
        result = dict(status="complete", setup_s=setup_s, elapsed_s=time.monotonic() - started,
                      frozen_weights_unchanged=True, chunks=count)
        analyze_case(out)
        # 解析と必須成果物まで揃ってから完了を確定する。
        save_json(out / "result.json", result)
        save_json(out / "progress.json", {**result, "completed_chunks": count, "total_chunks": count})
        save_json(out / "artifacts.json", inventory(out, exclude=("artifacts.json",)))
        return result
    finally:
        if sim.is_setup:
            model.unload()


def frozen_spikes(out, cfg):
    times, ids = [], []
    for path in sorted((out / "chunks").glob("[0-9]*.npz")):
        with np.load(path) as data:
            if str(data["phase"]) == "frozen":
                times.append(data["times_ms"] + float(data["start_ms"]) - cfg.development_ms)
                ids.append(data["ids"])
    if not times:
        raise ValueError("固定後の記録がありません")
    return np.concatenate(times), np.concatenate(ids)


def analyze_case(out):
    """固定後の連続観測全体を使い、チャンク端をアバランチ境界にしない。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.akita_soc.analysis.metrics import spike_columns

    out = Path(out)
    cfg = NetworkSettings.model_validate(json.loads((out / "settings.json").read_text()))
    times, ids = frozen_spikes(out, cfg)
    metrics = spike_columns(times, ids, cfg.n, cfg.frozen_ms, cfg.n)
    metrics = {k: None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v for k, v in metrics.items()}
    metrics["participants"] = int(np.unique(ids).size)
    metrics["interpretation"] = "記述指標。臨界性の確定ではなく、複数シード・時間・解析条件で継続評価する。"
    save_json(out / "frozen_metrics.json", metrics)
    with (out / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    x = np.array([float(r["end_ms"]) / 1000 for r in rows])
    for key in ("rate_exc_hz", "rate_inh_hz"):
        axes[0].plot(x, [float(r[key]) for r in rows], label=key)
    for key in ("mean_ee", "mean_ei", "mean_ie", "mean_ii"):
        axes[1].plot(x, [float(r[key]) if r[key] else np.nan for r in rows], label=key)
    axes[0].set_ylabel("Rate (Hz)")
    axes[1].set_ylabel("Mean weight")
    for ax in axes[:2]:
        ax.axvline(cfg.development_ms / 1000, color="black", linestyle="--", label="STDP off")
        ax.set_xlabel("Simulation time (s)")
        ax.legend()
    visible = times < min(cfg.frozen_ms, 10000)
    axes[2].scatter(times[visible] / 1000, ids[visible], s=0.4)
    axes[2].set(xlabel="Time after STDP off (s)", ylabel="Neuron ID")
    fig.tight_layout()
    fig.savefig(out / "activity.png", dpi=130)
    plt.close(fig)
    from src.utils.analysis.avalanche import split_avalanches
    from src.utils.analysis.powerlaw import discrete_distribution
    avalanche = split_avalanches(times)
    support, probability = discrete_distribution(avalanche.sizes)
    with np.load(out / "weights/frozen.npz") as frozen:
        weight = frozen["w"]
    with np.load(out / "connectivity.npz") as wiring:
        source_exc, target_exc = wiring["row"] < cfg.n_exc, wiring["col"] < cfg.n_exc
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for label, mask in zip(("EE", "EI", "IE", "II"),
                           (source_exc & target_exc, source_exc & ~target_exc,
                            ~source_exc & target_exc, ~source_exc & ~target_exc)):
        if mask.any():
            axes[0].hist(weight[mask], bins=np.linspace(0, 1, 31), histtype="step", density=True, label=label)
    axes[0].set(xlabel="Frozen weight", ylabel="Density")
    axes[0].legend()
    if support.size:
        axes[1].loglog(support, probability, ".")
    axes[1].set(xlabel="Avalanche size (spikes)", ylabel="Probability", title="Empirical distribution")
    bin_ms = min(10.0, cfg.chunk_ms)
    counts, edges = np.histogram(times, np.arange(0, cfg.frozen_ms + bin_ms, bin_ms))
    if counts.size > 2:
        power = np.abs(np.fft.rfft((counts - counts.mean()) * np.hanning(counts.size))) ** 2
        axes[2].plot(np.fft.rfftfreq(counts.size, bin_ms / 1000)[1:], power[1:])
    axes[2].set(xlabel="Frequency (Hz)", ylabel="Unnormalized power")
    fig.tight_layout()
    fig.savefig(out / "diagnostics.png", dpi=130)
    plt.close(fig)
    return metrics
