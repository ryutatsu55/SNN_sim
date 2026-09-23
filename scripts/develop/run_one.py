"""1 つの run を走らせる (子プロセス)。

    python -m scripts.develop.run_one <run ディレクトリ>

**引数は run ディレクトリ 1 つだけ。** そこに置かれた `config.yaml` が run の唯一の真実で、
seed も記録条件もすべてそこから読む。run ディレクトリを作るのも config.yaml を置くのも
ランチャ (`python -m scripts.develop`) の仕事。

失敗した seed はこれを直接叩けばやり直せる。run ディレクトリごとコピーして叩けば
同じ run になる。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config_manager import CONFIG_NAME, ConfigManager
from src.core.NetworkBuilder import NetworkBuilder
from src.core.layout import AXES_NAME
from src.models.network.connectors import AXONS_NAME
from src.core.simulator import GeNNSimulator
from scripts.develop.analysis import metrics
from scripts.develop.report import overview, panels, structure
from scripts.develop.store import paths
from src.utils.runview import BuiltNetwork as Built
from scripts.develop.store.records import (CONNECTIVITY_NAME, COORDS_NAME, METRICS_NAME,
                                           MS_PER_HOUR,
                                           SPIKES, TRACE, WEIGHTS,
                                           MetricsWriter, record_filename,
                                           save_connectivity, save_coords,
                                           save_spikes, save_trace,
                                           save_weight_values)
from scripts.develop.store.series import open_run

import src.models.neurons.akita_escape_lif
import src.models.neurons.akita_escape_lif_physical
import src.models.network.connectors
import src.models.network.delays
import src.models.network.space
import src.models.network.weights
import src.models.plasticity.custom_Akita
import src.models.synapses.standard_models
import src.models.synapses.custom



def parse_args():
    parser = argparse.ArgumentParser(
        description="run ディレクトリの config.yaml を読んで 1 run を走らせる。")
    parser.add_argument("run_dir", help="config.yaml が置かれた run ディレクトリ")
    return parser.parse_args()


def resolve_trace(config) -> tuple[int | None, float]:
    """膜電位トレースの設定を `task.trace_neuron` / `task.trace_window_s` から読む。

    Returns:
        `(ニューロン ID | None, 窓幅 [s])`。ID が None なら採らない (キーが無い場合も)。
    """
    neuron = getattr(config.task, "trace_neuron", None)
    window_s = float(getattr(config.task, "trace_window_s", 10.0))
    if neuron is None:
        return None, window_s
    neuron = int(neuron)
    total = int(config.simulation.N)
    if not 0 <= neuron < total:
        # 長い run を回し切ってから IndexError で落ちるのを防ぐ。
        raise SystemExit(
            f"task.trace_neuron={neuron} はニューロン数 {total} の範囲外です (0..{total - 1})。")
    if window_s <= 0:
        raise SystemExit(f"task.trace_window_s は正の値にしてください (got {window_s})。")
    return neuron, window_s


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
    """1 ステップずつ進めながら対象ニューロンの V と Isyn_rec を採り、窓内スパイクを返す。

    保持するのは対象ニューロン 1 本の列だけ。採取ステップ数は GeNN の記録バッファ長の
    整数倍に丸める (バッファはちょうど溜まった時しか読み出せないため)。

    Returns:
        `(V, I, spikes, actual_window_s)`。`actual_window_s` は丸めた後の実表示幅 [s]。
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


def _model_name(run_dir: Path, seed: int) -> str:
    """GeNN のモデル名。**並列実行する run どうしで必ず違う名前になること。**

    同じ名前だと `<model名>_CODE` を共有し、同時にコード生成した 2 プロセスが互いの
    生成物を壊す。
    """
    stem = run_dir.parent.name if run_dir.name.startswith("seed") else run_dir.name
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in stem)
    return f"develop_{safe}_seed{seed}"


def record_once(sim, series, run_dir: Path, hour: float, *,
                record_start_ms: float, record_window_steps: int, chunk_steps: int,
                trace_neuron: int | None, trace_window_s: float, metrics_csv) -> None:
    """記録時刻 1 点ぶん: 窓を走らせ、npz を書き、**書いたものを読み直して**出力する。

    最後の一手で `replot.py` とまったく同じ呼び出しになる。
    """
    config = series.config
    dt = float(config.simulation.dt)
    record_window_ms = float(config.task.record_window_ms)

    # 重みは窓に入る**前**の値。ここで pull しておかないと窓ぶんの可塑性が混ざる。
    weights = sim.pull_synapse_coo("w")["data"]
    save_weight_values(paths.data_path(run_dir, record_filename(WEIGHTS, hour)), weights)

    trace = None
    if trace_neuron is None:
        spikes = run_steps(sim, record_window_steps, chunk_steps, keep_spikes=True)
    else:
        # 記録窓の先頭を1ステップ刻みで進めて対象ニューロンの膜電位トレースを採取し、
        # 残りの窓は高速なチャンク実行で進める。両パートのスパイクを結合して窓全体の
        # スパイク列を得る。こうすることでラスター(窓全体)と neuron_trace(先頭)が
        # 同一タイミング・同一実現のデータとなり、ラスター先頭部分と直接比較できる。
        effective_trace_s = min(trace_window_s, record_window_ms / 1000.0)
        trace_v, trace_i, trace_spikes, trace_window_s = capture_membrane_window(
            sim, effective_trace_s, trace_neuron)
        trace_steps = len(trace_v)
        remaining_steps = record_window_steps - trace_steps
        if remaining_steps < 0:
            # ここを黙って通すと窓の総ステップ数がトレースの有無で変わり、結果が変わる。
            raise ValueError(
                f"トレース採取が記録窓を超えました ({trace_steps} > {record_window_steps} steps)。"
                " record_buffer_ms を record_window_ms より小さくしてください。"
            )
        rest = run_steps(sim, remaining_steps, chunk_steps, keep_spikes=True) if remaining_steps \
            else {"times": np.array([], dtype=np.float64), "ids": np.array([], dtype=np.int32)}
        times = np.concatenate([trace_spikes["times"], rest["times"]])
        ids = np.concatenate([trace_spikes["ids"], rest["ids"]])
        order = np.argsort(times)
        spikes = {"times": times[order], "ids": ids[order]}
        trace = (trace_v, trace_i, trace_spikes, trace_window_s)

    # **窓の原点を npz に埋める。** 再解析はこれを読むので、ファイル名から時刻を復元
    # しなくて済む (`f"{hour:g}"` は有効数字 6 桁なので非整数の記録時刻は往復しない)。
    save_spikes(paths.data_path(run_dir, record_filename(SPIKES, hour)),
                spikes["times"], spikes["ids"], record_start_ms)

    if trace is not None:
        trace_v, trace_i, trace_spikes, trace_window_s = trace
        # 図だけでなく生データも残す。1 ニューロン × 数万ステップで数百 KB しかない一方、
        # 「同じ窓のラスターと時間軸が揃った V/I」は後から作れない。
        save_trace(paths.data_path(run_dir, record_filename(TRACE, hour)),
                   trace_v, trace_i, dt=dt, neuron_id=trace_neuron,
                   window_s=trace_window_s,
                   spike_times=trace_spikes["times"] - record_start_ms,
                   spike_ids=trace_spikes["ids"])

    # 書いたものを読み直す。**ここから先は replot.py と同一の呼び出し。**
    panels.emit(series.window(hour), metrics=metrics_csv)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    # 引き継ぎ (`pending_config.yaml`) があればそれ、無ければ記録 (`config.yaml`)。
    # 両方が同時に存在することはない —— run 本体が config.yaml を書いた直後に引き継ぎを
    # 消すため。前者は「これから走る run」、後者は「完走した run をもう一度走らせる」。
    config_path = run_dir / paths.PENDING_CONFIG_NAME
    if not config_path.exists():
        config_path = run_dir / CONFIG_NAME
    if not config_path.exists():
        raise SystemExit(
            f"{run_dir} に {paths.PENDING_CONFIG_NAME} も {CONFIG_NAME} もありません。"
            " run ディレクトリを作って config を置くのは"
            " ランチャ (python -m scripts.develop) の仕事です。"
        )

    manager = ConfigManager()
    config = manager.load_resolved(config_path)
    seed = config.simulation.seed
    if not isinstance(seed, int):
        raise SystemExit(
            f"{config_path} の simulation.seed がスカラーではありません ({seed!r})。"
            " 範囲指定を展開するのはランチャの仕事です。"
        )

    trace_neuron, trace_window_s = resolve_trace(config)
    fig_kinds = paths.DEFAULT_FIG_KINDS + ((paths.TRACE,) if trace_neuron is not None else ())
    paths.prepare(run_dir, fig_kinds)

    print(f"run: {run_dir}  seed={seed}  backend={config.simulation.backend}")
    if trace_neuron is not None:
        print(f"  膜電位トレース: neuron {trace_neuron}, 窓の先頭 {trace_window_s} s")

    # コード生成先は `src/core/NetworkBuilder.py` の GENN_CODE_DIR (既定)。
    builder = NetworkBuilder(config, model_name=_model_name(run_dir, seed))
    genn_model, layout = builder.build(rec_spike=True)

    # **ここが「run の記録」を書く唯一の場所。** build() を通ったので network.sparse は
    # 実値 ("on"/"off") になっており、save_config() の不変条件を満たす。
    # 記録を書いたら引き継ぎファイルは役目を終えるので消す。以降この run ディレクトリには
    # config.yaml しか無く、それは「build を通った run の記録」であることが保証される。
    manager.save_config(config, save_dir=run_dir)
    (run_dir / paths.PENDING_CONFIG_NAME).unlink(missing_ok=True)

    # 外部軸 (layer / module など) は config だけからは復元できないので保存しておく。
    # 解析側は config.yaml から自動軸を再構築し、これを load_axes_file() で読み戻す。
    layout.save_axes(paths.data_path(run_dir, AXES_NAME))

    # 座標も config だけからは復元できない。**`no_space` では書かない** ——
    # 「ファイルが無い = 空間を持たない run」を読む側の判定にしているため。
    coords = builder.global_coords
    if coords is not None and np.all(np.isfinite(coords)):
        save_coords(paths.data_path(run_dir, COORDS_NAME), coords)

    # 軸索の折れ線は「どの軸索がどのブリッジを通ったか」= 損傷実験に要る記録。
    # 幾何を残すのは axon_growth 系のコネクタだけなので、持たないものは素通りさせる。
    geometry = getattr(builder.connection, "axon_geometry", lambda: None)()
    if geometry is not None:
        geometry.save(paths.data_path(run_dir, AXONS_NAME))
        print(f"  Saved axon geometry: {geometry.seg_owner.size} segments -> {AXONS_NAME}")

    # 構造図と数値レポート。シミュレーション結果には依存しないので、長い run が途中で
    # 落ちても構造の記録だけは残るよう **setup の前**に出す。
    print("\nGenerating network structure figures...")
    structure.emit(Built.from_builder(builder, run_dir, geometry=geometry))

    sim = GeNNSimulator(genn_model, config, builder)
    sim.setup()

    # 重み記録は常に COO の値だけ。結合構造はシミュレーション中に変わらないので
    # run につき 1 回だけ書く。row/col は pull_synapse_coo と同じ走査から得るので、
    # 値との並びが必ず一致する。
    connectivity = sim.synapse_connectivity_coo()
    weights_row, weights_col = connectivity["row"], connectivity["col"]
    save_connectivity(paths.data_path(run_dir, CONNECTIVITY_NAME),
                      weights_row, weights_col, connectivity["shape"])
    print(f"  Saved connectivity: {weights_row.size} synapses -> {CONNECTIVITY_NAME}")

    print(f"  Avalanche fit range: "
          f"[1, {metrics.resolve_avalanche_smax(config)}] (N={builder.total_neurons})")
    dt = float(config.simulation.dt)
    record_window_ms = float(config.task.record_window_ms)
    buffer_ms = float(getattr(config.task, "record_buffer_ms", record_window_ms))
    chunk_steps = max(1, int(buffer_ms / dt))
    record_window_steps = max(1, int(record_window_ms / dt))
    record_starts = sorted(float(hour) * MS_PER_HOUR for hour in config.task.record_hours)

    # **1 行ずつ書く。** 途中で落ちた run も、そこまでの指標が読める。
    metrics_csv = MetricsWriter(paths.data_path(run_dir, METRICS_NAME))
    # 記録はこれから書くので窓は 0 個。`series.window(hour)` が 1 窓ずつ拾っていく。
    series = open_run(run_dir, require_windows=False)
    current_ms = 0.0
    for record_start_ms in record_starts:
        develop_ms = record_start_ms - current_ms
        if develop_ms < -1e-9:
            raise ValueError("record_hours must be sorted and non-overlapping.")
        if develop_ms > 0:
            run_steps(sim, int(round(develop_ms / dt)), chunk_steps, keep_spikes=False)
            current_ms += develop_ms

        hour = record_start_ms / MS_PER_HOUR
        record_once(
            sim, series, run_dir, hour,
            record_start_ms=record_start_ms,
            record_window_steps=record_window_steps, chunk_steps=chunk_steps,
            trace_neuron=trace_neuron, trace_window_s=trace_window_s,
            metrics_csv=metrics_csv,
        )
        current_ms += record_window_steps * dt

    print("\nGenerating visualizations...")
    overview.emit(open_run(run_dir))

    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
