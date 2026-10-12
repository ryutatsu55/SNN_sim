"""1 つの損傷 run を走らせる (子プロセス)。

    python -m scripts.lesion.run_one <run ディレクトリ>

**引数は run ディレクトリ 1 つだけ。** そこに置かれた `config.yaml` が run の唯一の真実で、
親 run のパスも切断 spec も probe のタイムラインもすべて `task.*` から読む。run ディレクトリを
作るのも config を置くのもランチャ (`python -m scripts.lesion`) の仕事で、ここではやらない。

## 流れ

    下見 (build するが compile しない)
      → 親の重みを (pre, post) で復元し、切断対象を決める
      → 記録を書く (結合 / lesion.json / 切断プロファイル)
      → **ベースラインは親 run の記録時刻の窓をそのまま置く** (probe 0)
    本番 (compile する 1 回きりの build)
      → 復元した重みを載せ、切断マスクで行を削除
      → settle → record_hours の各時刻で probe

**ベースラインを自分で測らないのが要点。** 切断前のネットワークをもう一度走らせても、
settle と記録窓のあいだに起きるのは親 run の続きでしかない。測り直すぶん GeNN の
コンパイルが 2 回になるだけなので、親の `weights_{h}h.npz` と `spikes_{h}h.npz` を
そのままベースラインにする (引き継ぎ時刻は `task.from_hour`)。

**ベースライン窓の幅は親のもの**で、`task.record_window_ms` とは違いうる。窓幅は窓ごとに
npz が持つので、発火率は正しく出る (`store/views.py`)。

構造的除去が再ビルドを意味するのは GeNN の事情による —— シナプス集団は
`set_sparse_connections()` で確定し、実行中に行を削れない。切断は「生成直後の COO から
行を落としてから登録する」という形でしか書けない (`build_network()`)。
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config_manager import CONFIG_NAME, ConfigManager
from src.core.NetworkBuilder import GlobalCOO, NetworkBuilder
from src.core.simulator import GeNNSimulator
from src.utils.analysis.axons import synapse_crossed_bridges
from src.utils.analysis.connectivity import bridge_part_indices

from scripts.lesion.analysis import metrics as lesion_metrics
from src.utils.analysis.weights import align_saved_to_coo
from scripts.lesion.analysis.restore import (verify_axon_geometry,
                                             verify_geometry_alignment)
from scripts.lesion.analysis.selectors import LesionContext, combine, parse_cut_spec
from scripts.lesion.store.parent_run import load_parent
from scripts.lesion.report import overview, panels, structure
from scripts.lesion.store import paths
from scripts.lesion.store.built import write_built
from scripts.lesion.store.formats import (METRICS_NAME, MS_PER_HOUR,
                                          PHASE_BASELINE, PHASE_POST, SPIKES, TRACE,
                                          WEIGHTS, CUT_PROFILE_NAME, MetricsWriter,
                                          probe_filename, save_cut, save_manifest,
                                          save_spikes, save_trace,
                                          save_weight_values, write_table)
from scripts.lesion.store.formats import BASELINE_INDEX
from scripts.lesion.store.views import open_run

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
        description="run ディレクトリの config.yaml を読んで 1 つの損傷 run を走らせる。")
    parser.add_argument("run_dir", help="config.yaml が置かれた run ディレクトリ")
    parser.add_argument("--dry-run", action="store_true",
                        help="GeNN を触らず、切断対象を数えて lesion.json を書いて終了")
    return parser.parse_args()


def run_steps(sim, steps: int, chunk_steps: int, keep_spikes: bool):
    """chunk_steps ずつ進めながらスパイクを回収する。"""
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
        return {"times": np.array([], dtype=np.float32), "ids": np.array([], dtype=np.int32)}
    times = np.concatenate(all_times)
    ids = np.concatenate(all_ids)
    order = np.argsort(times)
    return {"times": times[order], "ids": ids[order]}


def capture_membrane_window(sim, window_s: float, neuron_id: int):
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


def build_network(config, model_name, restored_weights=None,
                  cut_mask=None, preserve_fan_in: bool = False, compile_model: bool = True):
    """再ビルド + (任意で) 重み復元と構造的除去。

    重みの差し替えと行の削除を**同じ 1 つの変換**で行う。`replace_global_coo()` は
    乱数の消費が終わった後・GeNN 登録の前に呼ばれるので、ネットワーク実現は
    同一 seed のまま変わらない。
    """
    # コード生成先は `src/core/NetworkBuilder.py` の GENN_CODE_DIR (既定)。
    builder = NetworkBuilder(config, model_name=model_name)

    def transform(coo: GlobalCOO) -> GlobalCOO:
        weights = coo.weights if restored_weights is None else np.asarray(restored_weights,
                                                                         dtype=np.float64)
        if cut_mask is None:
            return coo._replace(weights=weights)
        keep = ~np.asarray(cut_mask, dtype=bool)
        return GlobalCOO(row=coo.row[keep], col=coo.col[keep],
                         weights=weights[keep], delays=coo.delays[keep], shape=coo.shape)

    needs_transform = restored_weights is not None or cut_mask is not None
    if not compile_model:
        builder._generate_global_matrices()
        if needs_transform:
            builder.replace_global_coo(transform(builder.global_coo()),
                                       preserve_fan_in_scale=preserve_fan_in)
        return builder, None, builder.layout

    genn_model, layout = builder.build(
        rec_spike=True,
        transform_coo=transform if needs_transform else None,
        preserve_fan_in_scale=preserve_fan_in and needs_transform,
    )
    return builder, genn_model, layout


def probe(sim, series, run_dir: Path, index: int, *, start_ms: float,
          clock_start_ms: float, window_ms: float, chunk_steps: int, dt: float,
          trace_neuron: int | None, trace_window_s: float, metrics_csv) -> None:
    """probe 1 点ぶん: 窓を走らせ、npz を書き、**書いたものを読み直して**出力する。

    `start_ms` は窓の先頭の**切断軸**の時刻、`clock_start_ms` は同じ瞬間の **GeNN の
    時計**。settle のぶんだけずれるので、GeNN が返すスパイク時刻は切断軸へ移してから
    書く (npz の時刻は切断軸、`record_start_ms` はその上の原点)。

    最後の一手が要点。in-memory の値をそのまま図へ渡すと、本番と再解析で描画経路が
    2 本になる。一度書いてから `series.window(index)` で読み直せば、`replot.py` と
    まったく同じ呼び出しになる。

    切断**後**の窓しか通らない (ベースラインは `write_baseline()`)。

    `trace_neuron` が指定されていれば、**窓の先頭**を 1 ステップ刻みで進めて膜電位を
    採る。同じ窓のラスターと時間軸が揃うので、ラスターの先頭部分と直接見比べられる。
    """
    weights = sim.pull_synapse_coo("w")["data"]
    steps = max(1, int(round(window_ms / dt)))

    trace = None
    if trace_neuron is None:
        spikes = run_steps(sim, steps, chunk_steps, keep_spikes=True)
    else:
        effective_trace_s = min(trace_window_s, window_ms / 1000.0)
        trace_v, trace_i, trace_spikes, actual_window_s = capture_membrane_window(
            sim, effective_trace_s, trace_neuron)
        remaining = steps - len(trace_v)
        if remaining < 0:
            # 黙って通すと窓の総ステップ数がトレースの有無で変わり、結果が変わる。
            raise ValueError(
                f"トレース採取が記録窓を超えました ({len(trace_v)} > {steps} steps)。"
                " record_buffer_ms を record_window_ms より小さくしてください。")
        rest = (run_steps(sim, remaining, chunk_steps, keep_spikes=True) if remaining
                else {"times": np.array([], dtype=np.float64),
                      "ids": np.array([], dtype=np.int32)})
        times = np.concatenate([trace_spikes["times"], rest["times"]])
        ids = np.concatenate([trace_spikes["ids"], rest["ids"]])
        order = np.argsort(times)
        spikes = {"times": times[order], "ids": ids[order]}
        trace = (trace_v, trace_i, trace_spikes, actual_window_s)

    save_weight_values(paths.data_path(run_dir, probe_filename(WEIGHTS, index)), weights)
    # **窓の素性を npz に埋める。** 原点 (切断からの経過) も窓幅も phase もここが正で、
    # ファイル名の index は並べ替えと目印にしか使わない。
    # GeNN の時計 -> 切断軸。settle は切断軸に含まれないので、引かないと窓ローカル時刻が
    # settle_ms ぶん先へずれ、ラスターも burstiness のビン割りも窓の外を見る。
    to_axis = start_ms - clock_start_ms
    save_spikes(paths.data_path(run_dir, probe_filename(SPIKES, index)),
                np.asarray(spikes["times"], dtype=np.float64) + to_axis, spikes["ids"],
                record_start_ms=start_ms, record_window_ms=window_ms, phase=PHASE_POST)

    if trace is not None:
        trace_v, trace_i, trace_spikes, actual_window_s = trace
        # 図だけでなく生データも残す。1 ニューロン x 数万ステップで数百 KB しかない一方、
        # 「同じ窓のラスターと時間軸が揃った V/I」は後から作れない。
        save_trace(paths.data_path(run_dir, probe_filename(TRACE, index)),
                   trace_v, trace_i, dt=dt, neuron_id=trace_neuron,
                   window_s=actual_window_s,
                   spike_times=np.asarray(trace_spikes["times"], dtype=np.float64)
                   - clock_start_ms,
                   spike_ids=trace_spikes["ids"])

    # 書いたものを読み直す。**ここから先は replot.py と同一の呼び出し。**
    panels.emit(series.window(index), metrics=metrics_csv)


def write_baseline(series, run_dir: Path, parent, weights, metrics_csv) -> None:
    """親 run の記録時刻の窓を、この run のベースライン (probe 0) として置く。

    **測り直さない。** 切断前のネットワークをもう一度走らせても、settle と記録窓の
    あいだに起きるのは親 run の続きでしかないので、親の記録をそのまま使う。

    窓は切断より前なので**負の時刻**に置く。0.0 にすると post の最初の probe と重なり、
    切断の瞬間が図でも CSV でも判別できなくなる。幅は親の記録窓のもので、
    `task.record_window_ms` とは違いうる (窓幅は窓ごとに npz が持つ)。

    `weights` は親の値を**この run の COO の並びへ引き当てた**ベクトル。

    親が膜電位トレースを採っていれば、その npz も**そのまま**引き継ぐ (中の時刻は
    窓の先頭からの相対なので、窓の原点が変わっても読み替えが要らない)。
    """
    start_ms = -float(parent.window_ms)
    save_weight_values(paths.data_path(run_dir, probe_filename(WEIGHTS, BASELINE_INDEX)),
                       weights)
    # npz は絶対時刻で持ち、原点を `record_start_ms` に持つ (`load_spikes` が引く)。
    # 親から読んだ時刻は窓ローカルなので、この窓の原点を足してから書く。
    save_spikes(paths.data_path(run_dir, probe_filename(SPIKES, BASELINE_INDEX)),
                np.asarray(parent.spike_times, dtype=np.float64) + start_ms,
                parent.spike_ids,
                record_start_ms=start_ms, record_window_ms=float(parent.window_ms),
                phase=PHASE_BASELINE)
    if parent.trace_path is not None:
        shutil.copyfile(parent.trace_path,
                        paths.data_path(run_dir, probe_filename(TRACE, BASELINE_INDEX)))

    print(f"  ベースライン: 親 run の {parent.hour} h の窓 "
          f"({parent.spike_times.size} spikes / {parent.window_ms / 1000:.0f} s"
          f"{', トレースあり' if parent.trace_path is not None else ''})")

    panels.emit(series.window(BASELINE_INDEX), metrics=metrics_csv)


def select_cut(scout, layout, config, parent):
    """切断対象を決める。GeNN を触らないので `--dry-run` はここまで走る。"""
    protocol = config.task
    coo = scout.global_coo()
    restored = align_saved_to_coo(parent.wiring.row, parent.wiring.col, parent.weights, coo)
    print(f"  重み復元: {restored.size} 本 "
          f"(mean={restored.mean():.4f}, max={restored.max():.4f})")

    geometry = getattr(scout.connection, "axon_geometry", lambda: None)()
    verification = {"coo_set_identical": True}
    if geometry is not None:
        verify_geometry_alignment(geometry, coo)
        verification["geometry_aligned_with_coo"] = True
        if parent.geometry is not None:
            verify_axon_geometry(geometry, parent.geometry)
            verification["axon_geometry_identical"] = True
            print("  軸索幾何が親 run と一致することを確認しました")

    specs = list(getattr(protocol, "cut", []) or [])
    if not specs:
        raise SystemExit(
            "task.cut が空です。切断 spec を 1 つ以上書いてください"
            " (例: bridge:kind=inter_cluster)。")
    context = LesionContext(coo=coo, layout=layout, area=scout.area, geometry=geometry,
                            coords=scout.global_coords)
    selection = combine([parse_cut_spec(spec).select(context) for spec in specs],
                        str(getattr(protocol, "cut_combine", "or")))
    num_cut = int(selection.cut.sum())
    print(f"  切断: {num_cut} / {coo.row.size} 本 "
          f"({num_cut / coo.row.size * 100:.1f}%)  [{selection.label}]")
    if num_cut == 0:
        raise SystemExit("切断対象が 0 本です。spec を確認してください。")
    if num_cut == coo.row.size:
        raise SystemExit("全シナプスが切断対象です。下流の解析がすべて NaN になるので止めます。")
    return coo, restored, geometry, selection, verification


def cross_check(scout, layout, coo, geometry):
    """2 つの「モジュール間」の定義を両方数える。

    **一致しない。** 位相的な「module ラベルが違う」と幾何的な「ブリッジを横切った」は
    別の集合なので、両方残して解釈時に取り違えないようにする。

    Returns: (cross_check dict, crossed 行列 | None, ブリッジの part 名 | None)
    """
    result = {}
    if layout.has_axis("module"):
        labels = np.asarray(layout.labels("module"))
        result["between_module_topological"] = int(
            np.count_nonzero(labels[coo.row] != labels[coo.col]))
    crossed = crossed_names = None
    if geometry is not None and scout.area is not None:
        try:
            bridges = bridge_part_indices(scout.area)
            crossed = synapse_crossed_bridges(geometry, scout.area, bridges)
            crossed_names = [str(scout.area.part_names[int(b)]) for b in bridges]
            result["bridge_crossing_geometric"] = int(crossed.any(axis=1).sum())
        except ValueError:
            # ブリッジを持たない area (no_space など)。幾何的な集計はできないだけ。
            crossed = None
    return result, crossed, crossed_names


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    config_path = run_dir / CONFIG_NAME
    if not config_path.exists():
        raise SystemExit(
            f"{run_dir} に {CONFIG_NAME} がありません。"
            " run ディレクトリを作って config を置くのは"
            " ランチャ (python -m scripts.lesion) の仕事です。"
        )

    manager = ConfigManager()
    config = manager.load_resolved(config_path)
    protocol = config.task
    seed = config.simulation.seed
    if not isinstance(seed, int):
        raise SystemExit(
            f"{config_path} の simulation.seed がスカラーではありません ({seed!r})。")

    parent_run = getattr(protocol, "parent_run", None)
    if not parent_run:
        raise SystemExit(
            f"{config_path} の task.parent_run がありません。"
            " 引き継ぐ親 run を決めるのはランチャの仕事です。")
    parent = load_parent(Path(parent_run), getattr(protocol, "from_hour", None))

    dt = float(config.simulation.dt)
    window_ms = float(protocol.record_window_ms)
    # **基準は切断の瞬間。** ランチャが展開済みの時刻 [h] を並べてある。
    record_starts = sorted(float(hour) * MS_PER_HOUR for hour in protocol.record_hours)
    chunk_steps = max(1, int(float(protocol.record_buffer_ms) / dt))
    trace_neuron = getattr(protocol, "trace_neuron", None)
    trace_window_s = float(getattr(protocol, "trace_window_s", 10.0))
    smax = lesion_metrics.resolve_avalanche_smax(config)

    print(f"run: {run_dir}  seed={seed}  backend={config.simulation.backend}")
    print(f"親 run: {parent.run_dir}  引き継ぎ時刻: {parent.hour} h  "
          f"N={config.simulation.N}  smax={smax}")

    # --- 切断対象の決定 (GeNN 不要。ここまでが --dry-run の範囲) -------------------
    scout, _model, layout = build_network(config, "lesion_scout", compile_model=False)
    # **外部軸は切断対象を決める前に読み戻す。** 軸自体は build が貼り直すが、親の記録で
    # 釘を刺しておく。`between:axis=...` は軸を見て切るので、後から読むと「親とは違う
    # ラベルで切った」ことに気づけないまま通る。
    if parent.axes_path is not None:
        layout.load_axes_file(parent.axes_path)
    coo, restored, geometry, selection, verification = select_cut(scout, layout, config, parent)
    checks, crossed, crossed_names = cross_check(scout, layout, coo, geometry)

    # 切断したものの素性を、残存群と同じ土俵で記述する。ニューロン単位の指標は
    # **切断前**のネットワークで測る (「切った時点でそれがどういう位置にいたか」が
    # 知りたいことなので)。重みは config が生成した初期値ではなく復元した実値を使う。
    per_synapse, comparison, cut_summary = lesion_metrics.cut_profile(
        coo._replace(weights=restored), selection.cut, layout, scout.total_neurons,
        coords=scout.global_coords, crossed=crossed, crossed_part_names=crossed_names)
    print()
    print(lesion_metrics.format_cut_profile(comparison, cut_summary))
    print()

    num_cut = int(selection.cut.sum())
    manifest = {
        "parent_run": str(parent.run_dir.resolve()),
        "restored_from_hour": parent.hour,
        "cut_specs": list(getattr(protocol, "cut", []) or []),
        "cut_combine": str(getattr(protocol, "cut_combine", "or")),
        "label": selection.label,
        "description": selection.description,
        "num_synapses_before": int(coo.row.size),
        "num_synapses_cut": num_cut,
        "num_synapses_after": int(coo.row.size - num_cut),
        "selector_detail": selection.detail,
        "cut_profile": cut_summary,
        "cross_check": checks,
        "verification": verification,
        "avalanche_smax": smax,
        "preserve_fan_in_scale": True,
        "parent_drift": parent.drift,
        "baseline_from_parent": {
            "hour": parent.hour,
            "window_ms": float(parent.window_ms),
            "note": "ベースラインは測り直さず、親 run の記録時刻の窓をそのまま使う",
        },
        "timeline_ms": {
            # settle は時間軸に含まれない。record_hours の 0 h は settle 明けの時刻。
            "settle": float(protocol.settle_ms),
            "probe_window": window_ms,
            "record_starts": record_starts,
        },
    }

    if args.dry_run:
        # **記録は run ディレクトリへ書かない。** dry-run は「何本切れるか」を見るための
        # 下見で、走っていない run のディレクトリに記録を残すと、完走した run と
        # 見分けが付かなくなる。
        import json
        print(json.dumps({k: manifest[k] for k in
                          ("label", "num_synapses_before", "num_synapses_cut",
                           "num_synapses_after", "cross_check", "verification")},
                         ensure_ascii=False, indent=2))
        print(f"--dry-run: GeNN は実行していません。{run_dir}")
        return

    # ---------------- ここから先が「走る run」 ----------------
    # **dry-run はここへ来ないので、下見のために data/ や figures/ が作られることはない**
    # (作ると完走した run と見分けが付かない)。
    # trace は採るときだけディレクトリを作る (空ディレクトリが残ると「採取したのに
    # 空だった」のか「採取していない」のかが読めなくなる)。
    paths.prepare(run_dir,
                  paths.DEFAULT_FIG_KINDS
                  + ((paths.TRACE,) if trace_neuron is not None else ()))
    print(f"\n=== {num_cut} 本を構造的に除去して回復を追う ===")
    keep = ~np.asarray(selection.cut, dtype=bool)
    # **モデル名は同時に走るものどうしで必ず変える** (同じ名前は genn_code の同じ
    # ディレクトリを共有する。ルート CLAUDE.md)。seed と切断条件で分ける。
    builder, model, layout = build_network(
        config, f"lesion_seed{seed}_{selection.label}"[:80],
        restored_weights=restored, cut_mask=selection.cut, preserve_fan_in=True)
    if parent.axes_path is not None:
        layout.load_axes_file(parent.axes_path)

    # 切断直後 (= この run の build 直後) の記録。何を書くかは `store/built.py` の
    # 一覧。**軸索幾何はここで切断後へ絞って書く** (コネクタの幾何は
    # `replace_global_coo()` の影響を受けない)。
    write_built(run_dir, builder, layout, pre_coo=coo, keep=keep, geometry=geometry)
    save_manifest(paths.data_dir(run_dir), manifest)
    save_cut(paths.data_dir(run_dir), per_synapse)
    write_table(comparison, paths.data_path(run_dir, CUT_PROFILE_NAME))

    # **書いたら読み直す。** 構造図も probe ごとの図と同じく記録から描くので、
    # `replot.py` とまったく同じ呼び出しになる。probe はこれから書くので 0 個。
    series = open_run(run_dir, require_windows=False)
    print("\nGenerating post-lesion structure figures...")
    structure.emit(series.built())

    sim = GeNNSimulator(model, config, builder)
    sim.setup(backup_initial_states=False)   # 復元前の重みを initial_states に残さない

    # **復元経路の自己検証。** GeNN が実際に持っている重みが、親から引き当てた値の
    # 残存ぶんと一致することを確かめる。ここが狂うと以降の指標は全部別物になる。
    check = sim.pull_synapse_coo("w")
    aligned = align_saved_to_coo(check["row"], check["col"], check["data"],
                                 builder.global_coo())
    if not np.allclose(aligned, restored[keep], rtol=0, atol=1e-6):
        raise RuntimeError("復元した重みが GeNN 上の値と一致しません (復元経路の破損)。")
    print("  親 run の重みが残存シナプスへ正しく入ったことを確認しました")

    metrics_csv = MetricsWriter(paths.data_path(run_dir, METRICS_NAME))
    write_baseline(series, run_dir, parent, restored, metrics_csv)

    # **settle は時間軸に含まれない。** record_hours の 0 h は settle 明けの時刻。
    # そのぶん GeNN の時計は切断軸より先を指すので、ずれを持っておいて probe へ渡す。
    if protocol.settle_ms > 0:
        run_steps(sim, int(round(float(protocol.settle_ms) / dt)), chunk_steps,
                  keep_spikes=False)
    clock_offset_ms = float(protocol.settle_ms)

    elapsed_ms = 0.0
    for k, record_start_ms in enumerate(record_starts):
        gap_ms = record_start_ms - elapsed_ms
        if gap_ms < -1e-9:
            raise ValueError(
                f"record_hours の窓が重なっています ({record_start_ms / MS_PER_HOUR:.3f} h)。"
                " 間隔を record_window_ms より広げてください。")
        if gap_ms > 0:
            run_steps(sim, int(round(gap_ms / dt)), chunk_steps, keep_spikes=False)
            elapsed_ms += gap_ms
        # **番号は `record_hours` の添字そのもの。** ベースラインは別名なのでずらさない。
        index = k
        print(f"  probe {index + 1}/{len(record_starts)}"
              f"  t = {elapsed_ms / MS_PER_HOUR:.2f} h")
        probe(sim, series, run_dir, index, start_ms=elapsed_ms,
              clock_start_ms=elapsed_ms + clock_offset_ms, window_ms=window_ms,
              chunk_steps=chunk_steps, dt=dt, trace_neuron=trace_neuron,
              trace_window_s=trace_window_s, metrics_csv=metrics_csv)
        elapsed_ms += window_ms

    print("\nGenerating visualizations...")
    overview.emit(open_run(run_dir))

    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
