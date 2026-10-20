import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigManager
from src.core.layout import NetworkLayout
from src.core.NetworkBuilder import NetworkBuilder
from src.core.output_manager import (
    AXES_NAME,
    AXONS_NAME,
    CONFIG_NAME,
    CONNECTIVITY_NAME,
    create_run_output_dir,
    create_timestamped_output_dir,
    data_dir,
    locate,
    organize_output,
    require,
)
from src.core.simulator import GeNNSimulator
from src.utils.analysis.avalanche import split_avalanches
from src.utils.analysis.criticality import (
    bimodality_d,
    burstiness_index,
    criticality_index_delta_cr,
)
from src.utils.analysis.powerlaw import log_likelihood_ratio_power_vs_exponential
from src.utils.analysis.spikes import diagnose_activity, firing_rates, spike_group_metrics
from src.utils.analysis.weights import block_values, weight_block_metrics
from src.utils.experiments.akita_soc.fig2c import plot_figure2c
from src.utils.experiments.akita_soc.runio import (
    SPIKES,
    WEIGHTS,
    discover_records,
    record_filename,
)
from src.utils.experiments.akita_soc.fig2d import plot_figure2d
from src.utils.experiments.akita_soc.weight_track import visualize_weight_tracks
from src.utils.plotting.distributions import plot_avalanche_distribution
from src.utils.plotting.raster import plot_raster
from src.utils.plotting.traces import neuron_trace

import src.models.neurons.akita_escape_lif
import src.models.neurons.akita_escape_lif_physical
import src.models.network.connectors
import src.models.network.delays
import src.models.network.space
import src.models.network.weights
import src.models.plasticity.custom_Akita
import src.models.synapses.standard_models
import src.models.synapses.custom


TASK_NAME = "akita_soc_fig2"
REPLOT_METRICS_NAME = "metrics_replot.csv"
TRACE_WINDOW_S = 10.0
TRACE_NEURON_ID = 0
PAPER_RASTER_XLIM_S = (0.0, 30.0)
PAPER_RASTER_YLIM_NEURON = (0.0, 100.0)
PAPER_AVALANCHE_XLIM = (1.0, 1000.0)
PAPER_AVALANCHE_YLIM = (1e-5, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="AkitaDai APL 2023 Fig.2相当の代表条件を実行します。")
    parser.add_argument("--config", default="configs/akita_soc.yaml")
    parser.add_argument("--replot-from", default=None, help="既存の実験出力ディレクトリからPNGだけを再生成する")
    parser.add_argument("--duration-hours", type=float, default=None)
    parser.add_argument("--record-hours", type=float, nargs="*", default=None)
    parser.add_argument(
        "--record-hours-range",
        type=float,
        nargs="+",
        metavar=("START", "STOP"),
        default=None,
        help="START STOP [STEP=1] で等間隔の記録時刻リストを生成 (例: --record-hours-range 0 72 1)",
    )
    parser.add_argument("--record-window-ms", type=float, default=None)
    parser.add_argument("--record-buffer-ms", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None, help="日時付き実行ディレクトリを作るベースディレクトリ")
    parser.add_argument(
        "--genn-code-dir",
        default="genn_code",
        help="GeNN生成コード(<model名>_CODE)を集約する親ディレクトリ (既定: genn_code)",
    )
    parser.add_argument("--task-name", default=TASK_NAME)
    return parser.parse_args()


def apply_overrides(config, args):
    if args.duration_hours is not None:
        config.task.duration = args.duration_hours * 60.0 * 60.0 * 1000.0
    if args.record_hours is not None and len(args.record_hours) > 0:
        config.task.record_hours = args.record_hours
    if args.record_hours_range is not None:
        parts = args.record_hours_range
        start = parts[0]
        stop = parts[1] if len(parts) >= 2 else parts[0]
        step = parts[2] if len(parts) >= 3 else 1.0
        config.task.record_hours = list(np.arange(start, stop + step * 0.5, step))
    if args.record_window_ms is not None:
        config.task.record_window_ms = args.record_window_ms
    if args.record_buffer_ms is not None:
        config.task.record_buffer_ms = args.record_buffer_ms
    if args.seed is not None:
        config.simulation.seed = args.seed


def run_steps(sim: GeNNSimulator, steps: int, chunk_steps: int, keep_spikes: bool):
    all_times = []
    all_ids = []
    remaining = int(steps)
    while remaining > 0:
        n_steps = min(remaining, chunk_steps)
        sim.step(n_steps)
        if keep_spikes:
            spikes = sim.get_global_spikes()
            if spikes["times"].size > 0:
                all_times.append(spikes["times"])
                all_ids.append(spikes["ids"])
        sim.flush_recording()
        remaining -= n_steps

    if not all_times:
        return {
            "times": np.array([], dtype=np.float32),
            "ids": np.array([], dtype=np.int32),
        }
    times = np.concatenate(all_times)
    ids = np.concatenate(all_ids)
    order = np.argsort(times)
    return {"times": times[order], "ids": ids[order]}


def capture_membrane_window(sim: GeNNSimulator, window_s: float, neuron_id: int):
    """1ステップずつ進めながら対象ニューロンの V と Isyn_rec を記録し、窓内スパイクを返す。

    メモリ節約のため全ニューロン行列ではなく対象ニューロン1本の列だけ保持する。
    GeNN の記録バッファは max_timesteps ちょうど溜まった時のみ読み出せるため、
    採取ステップ数はバッファ長の整数倍に丸め、バッファ境界ごとにスパイクを回収する。
    直前に run_steps がバッファをフラッシュ済みなので、回収されるのはこの窓のスパイクだけ。

    Returns:
        (V, I, spikes, actual_window_s): actual_window_s はバッファ整数倍に丸めた実表示幅[s]。
    """
    dt = sim.dt
    buf = sim.max_timesteps
    requested_steps = int(round(window_s * 1000.0 / dt))
    n_buffers = max(1, round(requested_steps / buf))
    steps = n_buffers * buf
    actual_window_s = steps * dt / 1000.0

    V = np.empty(steps, dtype=np.float64)
    I = np.empty(steps, dtype=np.float64)
    all_times = []
    all_ids = []
    for i in range(steps):
        sim.step()
        V[i] = sim.pull("V")[neuron_id]
        I[i] = sim.pull("Isyn_rec")[neuron_id]
        if (i + 1) % buf == 0:
            chunk = sim.get_global_spikes()
            if chunk["times"].size > 0:
                all_times.append(chunk["times"])
                all_ids.append(chunk["ids"])
            sim.flush_recording()

    if all_times:
        spikes = {"times": np.concatenate(all_times), "ids": np.concatenate(all_ids)}
    else:
        spikes = {"times": np.array([], dtype=np.float64), "ids": np.array([], dtype=np.int32)}
    return V, I, spikes, actual_window_s


def resolve_output_dir(out_dir_arg: str | None, suffix: str | None = None) -> Path:
    if out_dir_arg:
        output_dir = create_timestamped_output_dir(out_dir_arg, suffix=suffix)
        # ディレクトリが実際に作成されたか確認
        if not output_dir.exists():
            raise RuntimeError(f"Failed to create output directory: {output_dir}")
        return output_dir
    return create_run_output_dir("akita_soc")


def get_group_ids(config, layout):
    # config は後方互換のため残置。興奮性/抑制性の分類は NetworkLayout に集約された。
    return layout.ids_by("polarity")


def max_plasticity_weight(config) -> float:
    wmax_values = []
    for syn_cfg in config.synapses.values():
        wmax = getattr(syn_cfg.plasticity, "Wmax", None)
        if wmax is not None:
            wmax_values.append(float(wmax))
    return max(wmax_values) if wmax_values else 1.0


def replot_existing_output(run_dir: Path) -> None:
    config_path = require(run_dir, CONFIG_NAME)
    manager = ConfigManager()
    config = manager.load_resolved(config_path)
    record_window_ms = float(config.task.record_window_ms)
    total_neurons = int(config.simulation.N)
    spike_files = discover_records(run_dir, SPIKES)
    if not spike_files:
        raise FileNotFoundError(f"No spikes_*h.npz files found in: {run_dir}")

    # ラスターをニューロングループ順に並べ替えるためのグループ割り当てを再構築する。
    # NetworkLayout.from_config は config.layout.assignment (と seed) から割当を決定論的に
    # 再構築するため、GeNN コンパイルなしでも本番実行と同一のグローバルインデックス割当が
    # 得られる。
    # config だけでは再導出できない外部軸 (layer / module …) は layout_axes.npz から読み戻す。
    layout = None
    try:
        layout = NetworkLayout.from_config(config)
        axes_path = locate(run_dir, AXES_NAME)
        if axes_path is not None:
            layout.load_axes_file(axes_path)
    except Exception as e:
        print(f"  Warning: could not reconstruct layout for grouped raster: {e}")

    metrics_rows = []
    for record in spike_files:
        hour = record.hour
        spikes_npz = np.load(record.path)
        times = spikes_npz["times"]
        ids = spikes_npz["ids"]
        record_start_ms = hour * 60.0 * 60.0 * 1000.0
        local_times = times - record_start_ms

        avalanche = split_avalanches(local_times)
        rates = firing_rates(ids, total_neurons, record_window_ms)
        metrics_rows.append(
            {
                "hour": hour,
                "num_spikes": int(times.size),
                "mean_rate_hz": float(np.mean(rates)),
                "avalanche_threshold_ms": avalanche.threshold_ms,
                "num_avalanches": int(avalanche.sizes.size),
                "llr": log_likelihood_ratio_power_vs_exponential(avalanche.sizes),
                "delta_cr": criticality_index_delta_cr(avalanche.sizes),
                "burstiness_index": burstiness_index(local_times, record_window_ms),
                "bimodality_d": bimodality_d(avalanche.sizes),
            }
        )

        plot_raster(
            local_times,
            ids,
            run_dir / f"raster_{hour:g}h.png",
            f"Raster {hour:g} h",
            xlim_s=PAPER_RASTER_XLIM_S,
            ylim_neuron=PAPER_RASTER_YLIM_NEURON,
            layout=layout,
        )
        plot_avalanche_distribution(
            avalanche.sizes,
            run_dir / f"avalanche_{hour:g}h.png",
            f"Avalanche distribution {hour:g} h",
            xlim=PAPER_AVALANCHE_XLIM,
            ylim=PAPER_AVALANCHE_YLIM,
        )

    # 再計算した指標は data/ (= fig2c が読む場所) に別名で置く。metrics.csv を上書きしないのは
    # そちらが重みブロック等スパイクからは再計算できない列を持つため。
    replot_metrics = data_dir(run_dir) / REPLOT_METRICS_NAME
    with open(replot_metrics, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    # 本番実行 (main) と同じ Figure 2c / 2d も再生成する。
    # fig2c は metrics.csv と weights_*h.npz を、fig2d は spikes_*h.npz と config.yaml を
    # run_dir から読む。いずれも欠けている場合は各関数が警告を出して安全に return する。
    print(f"\nGenerating visualizations...")
    try:
        print(f"  Figure 2c...")
        plot_figure2c(str(run_dir), layout, metrics_name=REPLOT_METRICS_NAME)
    except Exception as e:
        print(f"  Warning: Figure 2c generation failed: {e}")

    try:
        print(f"  Figure 2d...")
        plot_figure2d(str(run_dir), layout)
    except Exception as e:
        print(f"  Warning: Figure 2d generation failed: {e}")

    print(f"Akita SoC plots regenerated from: {run_dir}")


def main():
    args = parse_args()
    if args.replot_from is not None:
        replot_existing_output(Path(args.replot_from))
        return

    manager = ConfigManager()
    config = manager.resolve(args.config, args.task_name)
    apply_overrides(config, args)

    # seed をモデル名・出力ディレクトリ名に反映する。
    # → GeNNコード生成先 (<model名>_CODE) も出力先も seed ごとに分離され、
    #   同一 config を複数 seed で並列実行しても衝突しない。
    seed_tag = f"seed{config.simulation.seed}"
    out_dir = resolve_output_dir(args.out_dir, suffix=seed_tag)

    model_name = f"{Path(args.config).stem}_{seed_tag}"
    builder = NetworkBuilder(config, model_name=model_name, code_gen_dir=args.genn_code_dir)
    genn_model, layout = builder.build(rec_spike=True)

    # config.yaml (実 seed 入りの解決後 config) と source_config.yaml (入力の逐語コピー)。
    # **build() の後**に呼ぶこと。network.sparse は生成時に実値 ("on"/"off") へ焼き込まれる
    # ので、先に保存すると「どちらで走ったか」が記録に残らない。
    manager.save_config(config, save_dir=out_dir)

    # 外部軸 (layer / module など) は config だけからは復元できないので保存しておく。
    # 解析側は config.yaml から自動軸を再構築し、これを load_axes_file() で読み戻す。
    layout.save_axes(out_dir / AXES_NAME)

    # 軸索の折れ線は「どの軸索がどのブリッジを通ったか」= 将来の損傷実験に要る記録。
    # 幾何を残すのは axon_growth 系のコネクタだけなので、持たないものは素通りさせる
    # (area: no_space + constant_prob の既存 config はここで何もしない)。
    geometry = getattr(builder.connection, "axon_geometry", lambda: None)()
    if geometry is not None:
        geometry.save(out_dir / AXONS_NAME)
        print(f"  Saved axon geometry: {geometry.seg_owner.size} segments -> {AXONS_NAME}")

    sim = GeNNSimulator(genn_model, config, builder)
    sim.setup()

    # 重み記録は常に COO の値だけ。結合構造はシミュレーション中に変わらないので
    # run につき 1 回だけ書く。row/col は pull_synapse_coo と同じ走査から得るので、
    # 値との並びが必ず一致する。
    connectivity = sim.synapse_connectivity_coo()
    weights_row, weights_col = connectivity["row"], connectivity["col"]
    np.savez_compressed(
        out_dir / CONNECTIVITY_NAME,
        row=weights_row, col=weights_col, shape=connectivity["shape"],
    )
    print(f"  Saved connectivity: {weights_row.size} synapses -> {CONNECTIVITY_NAME}")

    group_ids = layout.ids_by("polarity")
    wmax = max_plasticity_weight(config)
    dt = float(config.simulation.dt)
    record_window_ms = float(config.task.record_window_ms)
    buffer_ms = float(getattr(config.task, "record_buffer_ms", record_window_ms))
    chunk_steps = max(1, int(buffer_ms / dt))
    record_window_steps = max(1, int(record_window_ms / dt))
    record_starts = sorted(float(hour) * 60.0 * 60.0 * 1000.0 for hour in config.task.record_hours)

    metrics_rows = []
    current_ms = 0.0
    for record_start_ms in record_starts:
        develop_ms = record_start_ms - current_ms
        if develop_ms < -1e-9:
            raise ValueError("record_hours must be sorted and non-overlapping.")
        if develop_ms > 0:
            run_steps(sim, int(round(develop_ms / dt)), chunk_steps, keep_spikes=False)
            current_ms += develop_ms

        hour = record_start_ms / (60.0 * 60.0 * 1000.0)

        # Ensure output directory exists before saving
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        # 値だけを COO で保存する (row/col は connectivity.npz に 1 回だけ)。
        # pull_synapse_coo は connectivity と同じ走査を使うので並びが必ず一致する。
        weights_path = out_dir / record_filename(WEIGHTS, hour)
        weights = sim.pull_synapse_coo("w")["data"]
        try:
            np.savez_compressed(weights_path, data=weights)
        except FileNotFoundError as e:
            print(f"Error saving {weights_path}: {e}")
            print(f"Output dir exists: {out_dir.exists()}, is_dir: {out_dir.is_dir()}")
            raise

        # 記録窓の先頭を1ステップ刻みで進めて対象ニューロンの膜電位トレースを採取し、
        # 残りの窓は高速なチャンク実行で進める。両パートのスパイクを結合して窓全体の
        # スパイク列を得る。こうすることでラスター(窓全体)と neuron_trace(先頭 TRACE_WINDOW_S 秒)が
        # 同一タイミング・同一実現のデータとなり、ラスター先頭部分と直接比較できる。
        effective_trace_s = min(TRACE_WINDOW_S, record_window_ms / 1000.0)
        trace_v, trace_i, trace_spikes, trace_window_s = capture_membrane_window(
            sim, effective_trace_s, TRACE_NEURON_ID
        )
        trace_steps = len(trace_v)
        remaining_steps = record_window_steps - trace_steps
        if remaining_steps > 0:
            rest_spikes = run_steps(sim, remaining_steps, chunk_steps, keep_spikes=True)
        else:
            rest_spikes = {
                "times": np.array([], dtype=np.float64),
                "ids": np.array([], dtype=np.int32),
            }
        times = np.concatenate([trace_spikes["times"], rest_spikes["times"]])
        ids = np.concatenate([trace_spikes["ids"], rest_spikes["ids"]])
        order = np.argsort(times)
        spikes = {"times": times[order], "ids": ids[order]}
        current_ms += record_window_steps * dt
        local_times = spikes["times"] - record_start_ms
        trace_local_times = trace_spikes["times"] - record_start_ms
        np.savez_compressed(out_dir / record_filename(SPIKES, hour),
                            times=spikes["times"], ids=spikes["ids"])

        avalanche = split_avalanches(local_times)
        rates = firing_rates(spikes["ids"], builder.total_neurons, record_window_ms)
        row = {
            "hour": hour,
            "num_spikes": int(spikes["times"].size),
            "mean_rate_hz": float(np.mean(rates)),
            "avalanche_threshold_ms": avalanche.threshold_ms,
            "num_avalanches": int(avalanche.sizes.size),
            "llr": log_likelihood_ratio_power_vs_exponential(avalanche.sizes),
            "delta_cr": criticality_index_delta_cr(avalanche.sizes),
            "burstiness_index": burstiness_index(local_times, record_window_ms),
            "bimodality_d": bimodality_d(avalanche.sizes),
        }
        row.update(
            spike_group_metrics(
                spikes["ids"],
                group_ids["excitatory"],
                group_ids["inhibitory"],
                record_window_ms,
            )
        )
        # O(nnz)。COO は実結合のみを持つので、結合の無い箇所の 0 は最初から混ざらない。
        blocks = block_values(weights, weights_row, weights_col, layout)
        row.update(weight_block_metrics(blocks, wmax=wmax))
        row.update(
            diagnose_activity(
                mean_rate_hz=row["mean_rate_hz"],
                weight_at_max_fraction=row["weight_at_max_fraction"],
            )
        )
        metrics_rows.append(row)

        plot_raster(
            local_times,
            spikes["ids"],
            out_dir / f"raster_{hour:g}h.png",
            f"Raster {hour:g} h",
            xlim_s=PAPER_RASTER_XLIM_S,
            ylim_neuron=PAPER_RASTER_YLIM_NEURON,
            layout=layout,
        )
        plot_avalanche_distribution(
            avalanche.sizes,
            out_dir / f"avalanche_{hour:g}h.png",
            f"Avalanche distribution {hour:g} h",
            xlim=PAPER_AVALANCHE_XLIM,
            ylim=PAPER_AVALANCHE_YLIM,
        )

        # neuron_test 相当の単一ニューロン膜電位トレース。
        # 記録窓の先頭 trace_window_s 秒ぶんを採取済みなので、ラスター先頭と時間軸が揃う。
        try:
            neuron_trace(
                trace_v,
                trace_i,
                trace_local_times,
                trace_spikes["ids"],
                dt=dt,
                id=TRACE_NEURON_ID,
                window_s=trace_window_s,
                title=f"neuron_trace_{hour:g}h",
                save_path=str(out_dir),
            )
        except Exception as e:
            print(f"  Warning: neuron trace generation failed at {hour:g}h: {e}")

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    # 可視化関数を呼び出し
    print(f"\nGenerating visualizations...")
    try:
        print(f"  Figure 2c...")
        plot_figure2c(str(out_dir), layout)
    except Exception as e:
        print(f"  Warning: Figure 2c generation failed: {e}")

    try:
        print(f"  Figure 2d...")
        plot_figure2d(str(out_dir), layout)
    except Exception as e:
        print(f"  Warning: Figure 2d generation failed: {e}")

    try:
        print(f"  Weight matrix tracks...")
        visualize_weight_tracks(out_dir, layout)
    except Exception as e:
        print(f"  Warning: Weight matrix visualization failed: {e}")

    # データを整理（data フォルダに npz, csv, config.yaml をまとめる）
    print(f"\nOrganizing output data...")
    try:
        organize_output(out_dir)
    except Exception as e:
        print(f"  Warning: Data organization failed: {e}")

    print(f"Akita SoC results saved to: {out_dir}")


if __name__ == "__main__":
    main()
