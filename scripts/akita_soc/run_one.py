"""1 つの run を走らせる (子プロセス)。

    python -m scripts.akita_soc.run_one <run ディレクトリ>

**引数は run ディレクトリ 1 つだけ。** そこに置かれた `config.yaml` が run の唯一の真実で、
seed も記録条件もすべてそこから読む。run ディレクトリを作るのも config.yaml を置くのも
ランチャ (`python -m scripts.akita_soc`) の仕事。

失敗した seed はこれを直接叩けばやり直せる。run ディレクトリごとコピーして叩けば
同じ run になる。

## 流れ

    build (結合を生成し、GeNN モデルを作る)
      → build 直後の記録を書く (`store/built.py`) → 構造図 (`report/structure.py`)
      → setup (GeNN のコンパイル)
      → record_hours の各時刻について:
           その時刻まで発達させる (スパイクは捨てる)
           → 記録窓を走らせて記録する (`record_once`) → 窓ごとの図と指標 (`report/panels.py`)
      → run 全体の図 (`report/overview.py`)

記録は `<run>/data/`、図と表は `<run>/report/` に出る。

**図はすべて、いったん書いた記録を読み直して描く。** だから描画部分は `replot.py` と
まったく同じ呼び出しになり、走らせた直後の図と再解析の図が食い違わない。
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
from src.core.simulator import GeNNSimulator
from scripts.akita_soc.analysis import metrics
from scripts.akita_soc.report import overview, panels, structure
from scripts.akita_soc.store import paths
from scripts.akita_soc.store.built import write_built
from scripts.akita_soc.store.formats import (METRICS_NAME, MS_PER_HOUR, SPIKES, VOLTAGE, CURRENT, WEIGHTS,
                                           MetricsWriter, record_filename, record_hour, save_spikes,
                                           save_neuron_timeseries, save_weight_values)
from scripts.akita_soc.store.views import open_run

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

    ID を指定した run は、記録窓の先頭でそのニューロンの V と Isyn_rec を採り、
    トレース図はそれを描く。

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


def run_steps(sim: GeNNSimulator, steps: int, keep_spikes: bool):
    """`steps` ステップ進め、その間のスパイクを `{"times", "ids"}` (時刻順) で返す。

    時刻は GeNN の時計 [ms]。`keep_spikes=False` ならスパイクは読まずに捨て、空を返す
    (記録しない発達区間を速く進めるため)。GeNN の記録バッファがあふれないよう、
    バッファ長 (`sim.max_timesteps`) ずつ区切って進める。
    """
    all_times = []
    all_ids = []
    remaining = int(steps)
    while remaining > 0:
        n_steps = min(remaining, sim.max_timesteps)
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


def capture_neuron_state(sim, window_s: float, neuron_ids: np.ndarray):
    """記録窓の先頭 `window_s` [s] を 1 ステップずつ進め、`neuron_ids` の膜電位 V と
    入力電流 Isyn_rec を毎ステップ採る。

    ステップ数は GeNN の記録バッファ長の整数倍に丸める (バッファはちょうど溜まった時しか
    読み出せないため)。

    Returns:
        `(V, I, spikes)`。V / I は (len(neuron_ids), ステップ数) で、行の並びは `neuron_ids`。
        spikes はその間の全ニューロンのスパイクで、時刻は GeNN の時計。
    """
    buf = sim.max_timesteps
    requested_steps = int(round(window_s * 1000.0 / sim.dt))
    steps = max(1, round(requested_steps / buf)) * buf

    V = np.empty((len(neuron_ids), steps), dtype=np.float64)
    I = np.empty((len(neuron_ids), steps), dtype=np.float64)
    all_times = []
    all_ids = []
    for i in range(steps):
        sim.step()
        V[:, i] = sim.pull("V")[neuron_ids]
        I[:, i] = sim.pull("Isyn_rec")[neuron_ids]
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
    return V, I, spikes


def _model_name(run_dir: Path, seed: int) -> str:
    """GeNN のモデル名。**並列実行する run どうしで必ず違う名前になること。**

    同じ名前だと `<model名>_CODE` を共有し、同時にコード生成した 2 プロセスが互いの
    生成物を壊す。run ディレクトリ名は名前を重複させないためだけに使い、意味は読み取らない。
    """
    stem = run_dir.parent.name if run_dir.name.startswith("seed") else run_dir.name
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in stem)
    return f"akita_soc_{safe}_seed{seed}"


def record_once(sim, series, run_dir: Path, hour: float, *,
                record_start_ms: float, record_window_steps: int,
                trace_neuron: int | None, trace_window_s: float,
                metrics_csv) -> None:
    """記録時刻 1 点ぶん: 窓を走らせ、npz を書き、それを読み直して図と指標を出す。

    `trace_neuron` を指定すると、**窓の先頭** `trace_window_s` [s] を 1 ステップ刻みで
    進めてそのニューロンの膜電位と入力電流を採り、残りは通常どおり進める。スパイクは
    両方をつないで窓全体の 1 本にするので、トレースとラスターは同じ窓・同じ実現になる。
    """
    config = series.config
    dt = float(config.simulation.dt)
    record_window_ms = float(config.task.record_window_ms)

    # 重みは窓に入る**前**の値。ここで pull しておかないと窓ぶんの可塑性が混ざる。
    weights = sim.pull_synapse_coo("w")["data"]
    save_weight_values(paths.data_path(run_dir, record_filename(WEIGHTS, hour)), weights)

    trace = None
    if trace_neuron is None:
        spikes = run_steps(sim, record_window_steps, keep_spikes=True)
    else:
        effective_trace_s = min(trace_window_s, record_window_ms / 1000.0)
        trace_v, trace_i, trace_spikes = capture_neuron_state(
            sim, effective_trace_s, np.array([trace_neuron]))
        trace_steps = len(trace_v)
        remaining_steps = record_window_steps - trace_steps
        if remaining_steps < 0:
            # ここを黙って通すと窓の総ステップ数がトレースの有無で変わり、結果が変わる。
            raise ValueError(
                f"トレース採取が記録窓を超えました ({trace_steps} > {record_window_steps} steps)。"
                " record_buffer_ms を record_window_ms より小さくしてください。"
            )
        rest = run_steps(sim, remaining_steps, keep_spikes=True) if remaining_steps \
            else {"times": np.array([], dtype=np.float64), "ids": np.array([], dtype=np.int32)}
        times = np.concatenate([trace_spikes["times"], rest["times"]])
        ids = np.concatenate([trace_spikes["ids"], rest["ids"]])
        order = np.argsort(times)
        spikes = {"times": times[order], "ids": ids[order]}
        trace = (trace_v, trace_i)

    # **窓の原点を npz に埋める。** 再解析はこれを読むので、ファイル名から時刻を復元
    # しなくて済む (`f"{hour:g}"` は有効数字 6 桁なので非整数の記録時刻は往復しない)。
    save_spikes(paths.data_path(run_dir, record_filename(SPIKES, hour)),
                spikes["times"], spikes["ids"], record_start_ms)

    if trace is not None:
        trace_v, trace_i = trace
        save_neuron_timeseries(paths.data_path(run_dir, record_filename(VOLTAGE, hour)),
                               trace_v, neuron_ids=[trace_neuron], dt=dt)
        save_neuron_timeseries(paths.data_path(run_dir, record_filename(CURRENT, hour)),
                               trace_i, neuron_ids=[trace_neuron], dt=dt)

    # 書いたものを読み直して出力する (replot.py と同じ呼び出し)。
    panels.emit(series.window(hour), metrics=metrics_csv)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    config_path = run_dir / CONFIG_NAME
    if not config_path.exists():
        raise SystemExit(
            f"{run_dir} に {CONFIG_NAME} がありません。"
            " run ディレクトリを作って config を置くのは"
            " ランチャ (python -m scripts.akita_soc) の仕事です。"
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
    report_kinds = paths.DEFAULT_REPORT_KINDS + ((paths.TRACE,) if trace_neuron is not None else ())
    paths.prepare(run_dir, report_kinds)

    print(f"run: {run_dir}  seed={seed}  backend={config.simulation.backend}")
    if trace_neuron is not None:
        print(f"  膜電位トレース: neuron {trace_neuron}, 窓の先頭 {trace_window_s} s")

    # コード生成先は `src/core/NetworkBuilder.py` の GENN_CODE_DIR (既定)。
    builder = NetworkBuilder(config, model_name=_model_name(run_dir, seed))
    genn_model, layout = builder.build(rec_spike=True)

    # build 直後の記録 (結合・遅延・初期重み・座標・軸索・外部軸)。何を書くかは
    # `store/built.py` の一覧。
    write_built(run_dir, builder)
    # 記録窓はこれから書くので 0 個。`series.window(hour)` が 1 窓ずつ拾っていく。
    series = open_run(run_dir, require_windows=False)

    # 構造図と数値レポート。シミュレーション結果には依存しないので、長い run が途中で
    # 落ちても構造の記録だけは残るよう **setup の前**に出す。
    print("\nGenerating network structure figures...")
    structure.emit(series.built())

    sim = GeNNSimulator(genn_model, config, builder)
    sim.setup()

    print(f"  Avalanche fit range: "
          f"[1, {metrics.resolve_avalanche_smax(config)}] (N={builder.total_neurons})")
    dt = float(config.simulation.dt)
    record_window_ms = float(config.task.record_window_ms)
    record_window_steps = max(1, int(record_window_ms / dt))
    record_starts = sorted(float(hour) * MS_PER_HOUR for hour in config.task.record_hours)

    # **1 行ずつ書く。** 途中で落ちた run も、そこまでの指標が読める。
    metrics_csv = MetricsWriter(paths.data_path(run_dir, METRICS_NAME))
    current_ms = 0.0
    for record_start_ms in record_starts:
        develop_ms = record_start_ms - current_ms
        if develop_ms < -1e-9:
            raise ValueError(
                f"record_hours の窓が重なっています ({record_start_ms / MS_PER_HOUR:.3f} h)。"
                " 間隔を record_window_ms より広げてください。")
        if develop_ms > 0:
            run_steps(sim, int(round(develop_ms / dt)), keep_spikes=False)
            current_ms += develop_ms

        hour = record_hour(record_start_ms)
        record_once(
            sim, series, run_dir, hour,
            record_start_ms=record_start_ms,
            record_window_steps=record_window_steps,
            trace_neuron=trace_neuron, trace_window_s=trace_window_s,
            metrics_csv=metrics_csv,
        )
        current_ms += record_window_steps * dt

    print("\nGenerating visualizations...")
    overview.emit(open_run(run_dir))

    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
