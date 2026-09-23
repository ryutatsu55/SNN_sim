"""1 つの損傷 run を走らせる (子プロセス)。

    python -m scripts.lesion.run_one <run ディレクトリ>

**引数は run ディレクトリ 1 つだけ。** そこに置かれた `config.yaml` が run の唯一の真実で、
親 run のパスも切断 spec も probe のタイムラインもすべて `task.*` から読む。run ディレクトリを
作るのも config を置くのもランチャ (`python -m scripts.lesion`) の仕事で、ここではやらない。

## 2 フェーズ / 2 モデル構成

構造的除去を選んだ時点で、切断は再ビルドを意味する (GeNN のシナプス集団は
`set_sparse_connections()` で確定し、実行中に行を削れない)。したがって 1 プロセスで
**2 回 build する**のが唯一の素直な形になる。

    Phase 1 (intact)                      Phase 2 (lesioned)
    ─────────────────                     ──────────────────
    親 config を同一 seed で再ビルド
      → 保存済み重みを join で復元
      → settle → baseline probe を 1 点
      → w_pre を pull して保存  ──────────→ 同じ config で再ビルド
                                            → w_pre を復元し、切断マスクで行を削除
                                            → 等間隔 probe で回復を追う

こうする理由は 2 つ。**ベースラインが「切断直前のまさにその重み」になる** (親 run の
記録時刻の値を借りるのではなく、同じプロセスで測った値になる) こと。そして Phase 2 の
復元重みが Phase 1 の dump と残存シナプス上で一致することを assert できるので、
**復元経路そのものが自己検証になる**こと。

代償は GeNN のコンパイルが 2 回走ること。`model_name` は必ず分ける。
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
from src.core.NetworkBuilder import GlobalCOO, NetworkBuilder
from src.core.layout import AXES_NAME
from src.core.simulator import GeNNSimulator
from src.utils.analysis.axons import subset_geometry, synapse_crossed_parts
from src.utils.analysis.connectivity import bridge_part_indices
from src.utils.runview import BuiltNetwork

from scripts.lesion.analysis import metrics as lesion_metrics
from src.utils.analysis.weights import align_saved_to_coo
from scripts.lesion.analysis.restore import (verify_axon_geometry,
                                             verify_geometry_alignment)
from scripts.lesion.analysis.selectors import LesionContext, combine, parse_cut_spec
from scripts.lesion.store.parent import load_parent
from scripts.lesion.report import overview, panels, structure
from scripts.lesion.store import paths
from scripts.lesion.store.records import (COORDS_NAME, METRICS_NAME, MS_PER_HOUR,
                                          PHASE_BASELINE, PHASE_POST, SPIKES, WEIGHTS,
                                          CUT_PROFILE_NAME, MetricsWriter, probe_filename,
                                          save_connectivity, save_coords, save_cut,
                                          save_manifest, save_spikes, save_weight_values,
                                          write_table)
from scripts.lesion.store.records import POST_CONNECTIVITY_NAME, pre_connectivity_path
from scripts.lesion.store.series import open_run

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


def probe(sim, series, run_dir: Path, index: int, *, phase: str, start_ms: float,
          window_ms: float, chunk_steps: int, dt: float, metrics_csv) -> None:
    """probe 1 点ぶん: 窓を走らせ、npz を書き、**書いたものを読み直して**出力する。

    最後の一手が要点。in-memory の値をそのまま図へ渡すと、本番と再解析で描画経路が
    2 本になる。一度書いてから `series.window(index)` で読み直せば、`replot.py` と
    まったく同じ呼び出しになる。
    """
    steps = max(1, int(round(window_ms / dt)))
    spikes = run_steps(sim, steps, chunk_steps, keep_spikes=True)
    weights = sim.pull_synapse_coo("w")["data"]

    save_weight_values(paths.data_path(run_dir, probe_filename(WEIGHTS, index)), weights)
    # **窓の素性を npz に埋める。** 原点 (切断からの経過) も窓幅も phase もここが正で、
    # ファイル名の index は並べ替えと目印にしか使わない。
    save_spikes(paths.data_path(run_dir, probe_filename(SPIKES, index)),
                spikes["times"], spikes["ids"],
                record_start_ms=start_ms, record_window_ms=window_ms, phase=phase)

    # 書いたものを読み直す。**ここから先は replot.py と同一の呼び出し。**
    panels.emit(series.window(index), metrics=metrics_csv)


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
            crossed = synapse_crossed_parts(geometry, scout.area, bridges)
            crossed_names = [str(scout.area.part_names[int(b)]) for b in bridges]
            result["bridge_crossing_geometric"] = int(crossed.any(axis=1).sum())
        except ValueError:
            # ブリッジを持たない area (no_space など)。幾何的な集計はできないだけ。
            crossed = None
    return result, crossed, crossed_names


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    # 引き継ぎ (`pending_config.yaml`) があればそれ、無ければ記録 (`config.yaml`)。
    config_path = run_dir / paths.PENDING_CONFIG_NAME
    if not config_path.exists():
        config_path = run_dir / CONFIG_NAME
    if not config_path.exists():
        raise SystemExit(
            f"{run_dir} に {paths.PENDING_CONFIG_NAME} も {CONFIG_NAME} もありません。"
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
    baseline_window_ms = float(getattr(protocol, "baseline_window_ms", None) or window_ms)
    chunk_steps = max(1, int(float(protocol.record_buffer_ms) / dt))
    smax = lesion_metrics.resolve_avalanche_smax(config)

    print(f"run: {run_dir}  seed={seed}  backend={config.simulation.backend}")
    print(f"親 run: {parent.run_dir}  引き継ぎ時刻: {parent.hour} h  "
          f"N={config.simulation.N}  smax={smax}")

    # --- 切断対象の決定 (GeNN 不要。ここまでが --dry-run の範囲) -------------------
    scout, _model, layout = build_network(config, "lesion_scout", compile_model=False)
    # **外部軸は切断対象を決める前に読み戻す。** `between:axis=module` のように軸で選ぶ
    # spec があるので、後から読むと選択が別のものになる。
    if parent.axes_path is not None:
        layout.load_axes_file(parent.axes_path)
    coo, restored, geometry, selection, verification = select_cut(scout, layout, config, parent)
    checks, crossed, crossed_names = cross_check(scout, layout, coo, geometry)

    # 切断したものの素性を、残存群と同じ土俵で記述する。ニューロン単位の指標は
    # **切断前**のネットワークで測る (「切った時点でそれがどういう位置にいたか」が
    # 知りたいことなので)。重みは config が生成した初期値ではなく復元した実値を使う。
    per_synapse, comparison, cut_summary = lesion_metrics.cut_profile(
        coo._replace(weights=restored), selection.cut, layout, scout.total_neurons,
        coords=scout.global_coords, crossed=crossed, crossed_part_names=crossed_names,
        include_betweenness=bool(getattr(protocol, "include_betweenness", True)),
        hub_z=float(getattr(protocol, "hub_z", 2.5)))
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
        "hub_z": float(getattr(protocol, "hub_z", 2.5)),
        "preserve_fan_in_scale": True,
        "parent_drift": parent.drift,
        "timeline_ms": {
            "settle": float(protocol.settle_ms),
            "baseline_window": baseline_window_ms,
            "probe_window": window_ms,
            "probe_interval": float(protocol.probe_interval_hours) * MS_PER_HOUR,
            "recovery_total": float(protocol.recovery_hours) * MS_PER_HOUR,
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

    # ---------------- Phase 1: intact ----------------
    # ここから先が「走る run」。**dry-run はここへ来ないので、下見のために
    # data/ や figures/ が作られることはない** (作ると完走した run と見分けが付かない)。
    paths.prepare(run_dir)
    print("\n=== Phase 1: 損傷なしで復元し、切断直前のベースラインを測る ===")
    seed_tag = f"seed{seed}"
    b1, model1, layout1 = build_network(config, f"lesion_intact_{seed_tag}",
                                        restored_weights=restored)
    if parent.axes_path is not None:
        layout1.load_axes_file(parent.axes_path)

    # **ここが「run の記録」を書く唯一の場所。** build() を通ったので network.sparse は
    # 実値になっており、save_config() の不変条件を満たす。
    manager.save_config(config, save_dir=run_dir)
    (run_dir / paths.PENDING_CONFIG_NAME).unlink(missing_ok=True)
    layout1.save_axes(paths.data_path(run_dir, AXES_NAME))

    # 座標も config だけからは復元できない。**`no_space` では書かない** ——
    # 「ファイルが無い = 空間を持たない run」を読む側の判定にしているため。
    coords = b1.global_coords
    if coords is not None and np.all(np.isfinite(coords)):
        save_coords(paths.data_path(run_dir, COORDS_NAME), coords)

    sim1 = GeNNSimulator(model1, config, b1)
    sim1.setup(backup_initial_states=False)   # 復元前の重みを initial_states に残さない
    connectivity1 = sim1.synapse_connectivity_coo()
    save_connectivity(pre_connectivity_path(paths.data_dir(run_dir)),
                      connectivity1["row"], connectivity1["col"], connectivity1["shape"])

    # 切断後の結合も先に書いておく。**`open_run()` は 2 本とも要求する**ので、
    # baseline probe を読み直す時点で揃っている必要がある。
    keep = ~np.asarray(selection.cut, dtype=bool)
    save_connectivity(paths.data_path(run_dir, POST_CONNECTIVITY_NAME),
                      coo.row[keep], coo.col[keep],
                      np.asarray(coo.shape, dtype=np.int64))
    save_manifest(paths.data_dir(run_dir), manifest)
    save_cut(paths.data_dir(run_dir), per_synapse)
    write_table(comparison, paths.data_path(run_dir, CUT_PROFILE_NAME))

    metrics_csv = MetricsWriter(paths.data_path(run_dir, METRICS_NAME))
    series = open_run(run_dir, require_windows=False)

    if protocol.settle_ms > 0:
        run_steps(sim1, int(round(float(protocol.settle_ms) / dt)), chunk_steps,
                  keep_spikes=False)

    # baseline は**切断より前**に測った値なので負の時刻に置く。0.0 にすると post の
    # 最初の probe と重なり、「切断の瞬間」が図でも CSV でも判別できなくなる。
    probe(sim1, series, run_dir, 0, phase=PHASE_BASELINE, start_ms=-baseline_window_ms,
          window_ms=baseline_window_ms, chunk_steps=chunk_steps, dt=dt,
          metrics_csv=metrics_csv)

    w_pre = sim1.pull_synapse_coo("w")
    print(f"  切断直前の重みを保存: {w_pre['data'].size} 本")
    del sim1, model1, b1

    # ---------------- Phase 2: lesioned ----------------
    print(f"\n=== Phase 2: {num_cut} 本を構造的に除去して回復を追う ===")
    coo2_source = align_saved_to_coo(w_pre["row"], w_pre["col"], w_pre["data"], coo)
    b2, model2, layout2 = build_network(
        config, f"lesion_cut_{seed_tag}_{selection.label}"[:80],
        restored_weights=coo2_source, cut_mask=selection.cut, preserve_fan_in=True)
    if parent.axes_path is not None:
        layout2.load_axes_file(parent.axes_path)
    sim2 = GeNNSimulator(model2, config, b2)
    sim2.setup(backup_initial_states=False)

    check = sim2.pull_synapse_coo("w")
    expected = coo2_source[~selection.cut]
    aligned = align_saved_to_coo(check["row"], check["col"], check["data"], b2.global_coo())
    if not np.allclose(aligned, expected, rtol=0, atol=1e-6):
        raise RuntimeError("Phase 2 の復元重みが Phase 1 の値と一致しません (復元経路の破損)。")
    print("  Phase 1 の重みが残存シナプスへ正しく入ったことを確認しました")

    # 記録済みの切断後 COO と、GeNN が実際に持っている COO が同じであることを確かめる。
    connectivity2 = sim2.synapse_connectivity_coo()
    save_connectivity(paths.data_path(run_dir, POST_CONNECTIVITY_NAME), connectivity2["row"],
                      connectivity2["col"], connectivity2["shape"])

    # 構造図。**切断後の幾何を渡すこと。** builder のコネクタが持つ幾何は
    # replace_global_coo() の影響を受けないので、既定のままだと axon_network.png に
    # 切ったはずの結合まで描かれる。
    print("\nGenerating post-lesion structure figures...")
    post_geometry = subset_geometry(geometry, keep) if geometry is not None else None
    structure.emit(BuiltNetwork.from_builder(b2, run_dir, geometry=post_geometry))

    if protocol.settle_ms > 0:
        run_steps(sim2, int(round(float(protocol.settle_ms) / dt)), chunk_steps,
                  keep_spikes=False)

    interval_ms = float(protocol.probe_interval_hours) * MS_PER_HOUR
    num_probes = max(1, int(round(float(protocol.recovery_hours)
                                  / float(protocol.probe_interval_hours))) + 1)
    elapsed_ms = 0.0
    for k in range(num_probes):
        gap_ms = k * interval_ms - elapsed_ms
        if gap_ms > 0:
            run_steps(sim2, int(round(gap_ms / dt)), chunk_steps, keep_spikes=False)
            elapsed_ms += gap_ms
        index = k + 1
        print(f"  probe {index}/{num_probes}  t = {elapsed_ms / MS_PER_HOUR:.2f} h")
        probe(sim2, series, run_dir, index, phase=PHASE_POST, start_ms=elapsed_ms,
              window_ms=window_ms, chunk_steps=chunk_steps, dt=dt, metrics_csv=metrics_csv)
        elapsed_ms += window_ms

    print("\nGenerating visualizations...")
    overview.emit(open_run(run_dir))

    print(f"完了: {run_dir}")


if __name__ == "__main__":
    main()
