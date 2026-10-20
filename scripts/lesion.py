"""損傷 (lesion) 実験。

`scripts/develop.py` が育てたネットワークを引き継ぎ、**特定の接続を構造的に除去**して、
切断直前から回復までを等間隔で記録する。

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
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigManager
from src.core.NetworkBuilder import NetworkBuilder, GlobalCOO
from src.core.output_manager import (
    AXES_NAME,
    AXONS_NAME,
    CONFIG_NAME,
    CONNECTIVITY_NAME,
    create_timestamped_output_dir,
    data_dir,
    locate,
    organize_output,
    require,
)
from src.core.simulator import GeNNSimulator
from src.utils.analysis.avalanche import split_avalanches
from src.utils.analysis.axons import subset_geometry, synapse_crossed_parts
from src.utils.analysis.connectivity import bridge_part_indices
from src.utils.experiments.akita_soc.runio import (
    WEIGHTS as AKITA_WEIGHTS,
    discover_records,
    load_connectivity,
    load_weight_values,
)
from src.utils.experiments.lesion import figures as lesion_figures
from src.utils.experiments.lesion import metrics as lesion_metrics
from src.utils.experiments.lesion import runio
from src.utils.experiments.lesion.restore import (
    align_saved_to_coo,
    verify_axon_geometry,
    verify_geometry_alignment,
)
from src.utils.experiments.lesion.selectors import LesionContext, combine, parse_cut_spec
from src.utils.plotting.distributions import plot_avalanche_distribution
from src.utils.plotting.raster import plot_raster

# develop.py と共通の実行ヘルパ。ここは「発達ループを回す」という同じ仕事なので、
# コピーではなく import で共有する (develop.py 本体は一切変更しない)。
from scripts.develop import (
    PAPER_AVALANCHE_XLIM,
    PAPER_AVALANCHE_YLIM,
    PAPER_RASTER_XLIM_S,
    raster_ylim,
    resolve_avalanche_smax,
    resolve_order_axes,
    run_steps,
)
from scripts.visualize_network_structure import visualize_structure

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
HOUR_MS = 60.0 * 60.0 * 1000.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="develop.py の結果ネットワークを引き継ぎ、接続を切断して回復を追う。")
    parser.add_argument("parent_run", help="引き継ぐ元の run ディレクトリ")
    parser.add_argument("--from-hour", type=float, default=None,
                        help="引き継ぐ記録時刻 [h]。既定は親 run の最終記録")
    parser.add_argument("--cut", action="append", default=[], metavar="SPEC",
                        help="切断 spec。繰り返し可。例: bridge:kind=inter_cluster / "
                             "hub:metric=participation,top=3,direction=out / "
                             "between:axis=module / synapses:pairs=3-7+12-40")
    parser.add_argument("--cut-combine", choices=("or", "and"), default="or")
    parser.add_argument("--settle-ms", type=float, default=60000.0,
                        help="復元直後・切断直後に落ち着かせる時間 [ms]")
    parser.add_argument("--baseline-window-ms", type=float, default=None,
                        help="ベースライン記録窓 [ms]。既定は --probe-window-ms と同じ")
    parser.add_argument("--recovery-hours", type=float, default=12.0,
                        help="切断後に観察する総時間 [h]")
    parser.add_argument("--probe-interval-hours", type=float, default=1.0,
                        help="probe の間隔 [h] (等間隔)")
    parser.add_argument("--probe-window-ms", type=float, default=600000.0)
    parser.add_argument("--record-buffer-ms", type=float, default=10000.0)
    parser.add_argument("--avalanche-smax", type=int, default=None,
                        help="べき乗フィット / ΔCr の上限。既定は simulation.N")
    parser.add_argument("--hub-z", type=float, default=2.5,
                        help="ハブ判定の within-module z 閾値 (Guimera-Amaral の既定 2.5)。"
                             "この値は代謝ネットワーク由来なので、次数の小さい網では"
                             "下げないとハブが 0 個になることがある")
    parser.add_argument("--no-betweenness", action="store_true",
                        help="媒介中心性を計算しない (大きな N で重い)")
    parser.add_argument("--no-clustering", action="store_true")
    parser.add_argument("--no-structure-figures", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="GeNN を触らず、切断対象を数えて lesion.json を書いて終了")
    parser.add_argument("--out-dir", default="outputs/lesion")
    parser.add_argument("--genn-code-dir", default="genn_code")
    parser.add_argument("--task-name", default=TASK_NAME)
    return parser.parse_args()


def load_parent(run_dir: Path, from_hour: float | None):
    """親 run の config / 重み / 幾何を読む。"""
    config = ConfigManager().load_resolved(require(run_dir, CONFIG_NAME))
    source = data_dir(run_dir)
    records = discover_records(source, AKITA_WEIGHTS)
    if not records:
        raise FileNotFoundError(f"weights_*h.npz が見つかりません: {source}")
    if from_hour is None:
        record = records[-1]
    else:
        matches = [r for r in records if abs(r.hour - from_hour) < 1e-9]
        if not matches:
            available = [r.hour for r in records]
            raise FileNotFoundError(f"{from_hour} h の重み記録がありません (ある時刻: {available})")
        record = matches[0]
    connectivity = load_connectivity(source)
    values = load_weight_values(record.path)
    return config, record.hour, connectivity, values


def build_network(config, model_name, code_gen_dir, restored_weights=None,
                  cut_mask=None, preserve_fan_in: bool = False, compile_model: bool = True):
    """再ビルド + (任意で) 重み復元と構造的除去。

    重みの差し替えと行の削除を**同じ 1 つの変換**で行う。`replace_global_coo()` は
    乱数の消費が終わった後・GeNN 登録の前に呼ばれるので、ネットワーク実現は
    同一 seed のまま変わらない。
    """
    builder = NetworkBuilder(config, model_name=model_name, code_gen_dir=code_gen_dir)

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


def probe(sim, builder, layout, config, out_dir, index, phase, t_since_cut_ms,
          window_ms, chunk_steps, smax, args):
    """1 probe: 窓ぶん走らせてスパイクと重みを記録し、指標を返す。"""
    dt = float(config.simulation.dt)
    steps = max(1, int(round(window_ms / dt)))
    spikes = run_steps(sim, steps, chunk_steps, keep_spikes=True)

    coo = sim.synapse_connectivity_coo()
    weights = sim.pull_synapse_coo("w")["data"]
    np.savez_compressed(out_dir / runio.probe_filename(runio.SPIKES, index),
                        times=spikes["times"], ids=spikes["ids"])
    np.savez_compressed(out_dir / runio.probe_filename(runio.WEIGHTS, index), data=weights)

    local_times = spikes["times"] - (spikes["times"].min() if spikes["times"].size else 0.0)
    row = {"probe": index, "phase": phase,
           "hours_since_cut": t_since_cut_ms / HOUR_MS,
           "window_ms": window_ms}
    row.update(lesion_metrics.spike_metrics(
        local_times, spikes["ids"], layout, builder.total_neurons, window_ms,
        smax=smax, dt_ms=dt))
    row.update(lesion_metrics.structure_metrics(
        coo["row"], coo["col"], weights, layout, builder.total_neurons,
        include_betweenness=not args.no_betweenness,
        include_clustering=not args.no_clustering, hub_z=args.hub_z))
    lesion_metrics.add_diagnosis(row)

    # 図の失敗で長い run を落とさない (develop.py と同じ扱い)。
    order_axes = resolve_order_axes(layout)
    tag = f"p{index:03d}_{phase}"
    title = f"{phase} ({row['hours_since_cut']:+.2f} h)"
    try:
        plot_raster(local_times, spikes["ids"], out_dir / f"raster_{tag}.png",
                    f"Raster {title}", xlim_s=PAPER_RASTER_XLIM_S,
                    ylim_neuron=raster_ylim(builder.total_neurons),
                    layout=layout, order_axes=order_axes)
    except Exception as error:
        print(f"  Warning: ラスターの生成に失敗しました ({tag}): {error}")
    try:
        plot_avalanche_distribution(
            split_avalanches(local_times).sizes, out_dir / f"avalanche_{tag}.png",
            f"Avalanche {title}", xlim=PAPER_AVALANCHE_XLIM, ylim=PAPER_AVALANCHE_YLIM,
            fit_smax=smax)
    except Exception as error:
        print(f"  Warning: アバランチ図の生成に失敗しました ({tag}): {error}")
    return row


def main():
    args = parse_args()
    parent_run = Path(args.parent_run)
    config, from_hour, connectivity, saved_values = load_parent(parent_run, args.from_hour)
    smax = resolve_avalanche_smax(config, args.avalanche_smax)
    print(f"親 run: {parent_run}  引き継ぎ時刻: {from_hour} h  "
          f"N={config.simulation.N}  seed={config.simulation.seed}  smax={smax}")

    # --- 切断対象の決定 (GeNN 不要。ここまでが --dry-run の範囲) -------------------
    scout, _model, layout = build_network(config, "lesion_scout", None, compile_model=False)
    coo = scout.global_coo()
    restored = align_saved_to_coo(connectivity.row, connectivity.col, saved_values, coo)
    print(f"  重み復元: {restored.size} 本 (mean={restored.mean():.4f}, max={restored.max():.4f})")

    geometry = getattr(scout.connection, "axon_geometry", lambda: None)()
    verification = {"coo_set_identical": True}
    saved_geo_path = locate(parent_run, AXONS_NAME)
    if geometry is not None:
        verify_geometry_alignment(geometry, coo)
        verification["geometry_aligned_with_coo"] = True
        if saved_geo_path is not None:
            from src.models.network.connectors import AxonGeometry
            verify_axon_geometry(geometry, AxonGeometry.load(saved_geo_path))
            verification["axon_geometry_identical"] = True
            print("  軸索幾何が親 run と一致することを確認しました")

    axes_path = locate(parent_run, AXES_NAME)
    if axes_path is not None:
        layout.load_axes_file(axes_path)

    if not args.cut:
        raise SystemExit("--cut を 1 つ以上指定してください (例: --cut bridge:kind=inter_cluster)")
    ctx = LesionContext(coo=coo, layout=layout, area=scout.area, geometry=geometry,
                        coords=scout.global_coords)
    selection = combine([parse_cut_spec(spec).select(ctx) for spec in args.cut],
                        args.cut_combine)
    num_cut = int(selection.cut.sum())
    print(f"  切断: {num_cut} / {coo.row.size} 本 ({num_cut / coo.row.size * 100:.1f}%)"
          f"  [{selection.label}]")
    if num_cut == 0:
        raise SystemExit("切断対象が 0 本です。spec を確認してください。")
    if num_cut == coo.row.size:
        raise SystemExit("全シナプスが切断対象です。下流の解析がすべて NaN になるので止めます。")

    # 2 つの「モジュール間」の定義は一致しない。両方を残して解釈時に取り違えないようにする。
    cross_check = {}
    if layout.has_axis("module"):
        labels = np.asarray(layout.labels("module"))
        cross_check["between_module_topological"] = int(
            np.count_nonzero(labels[coo.row] != labels[coo.col]))
    crossed = None
    crossed_names = None
    if geometry is not None and scout.area is not None:
        try:
            bridges = bridge_part_indices(scout.area)
            crossed = synapse_crossed_parts(geometry, scout.area, bridges)
            crossed_names = [str(scout.area.part_names[int(b)]) for b in bridges]
            cross_check["bridge_crossing_geometric"] = int(crossed.any(axis=1).sum())
        except ValueError:
            # ブリッジを持たない area (no_space など)。幾何的な集計はできないだけ。
            crossed = None

    # 切断したものの素性を、残存群と同じ土俵で記述する。ニューロン単位の指標は
    # **切断前**のネットワークで測る (「切った時点でそれがどういう位置にいたか」が
    # 知りたいことなので)。重みは config が生成した初期値ではなく復元した実値を使う。
    profile_coo = coo._replace(weights=restored)
    per_synapse, comparison, cut_summary = lesion_metrics.cut_profile(
        profile_coo, selection.cut, layout, scout.total_neurons,
        coords=scout.global_coords, crossed=crossed, crossed_part_names=crossed_names,
        include_betweenness=not args.no_betweenness, hub_z=args.hub_z)
    print()
    print(lesion_metrics.format_cut_profile(comparison, cut_summary))
    print()

    out_dir = create_timestamped_output_dir(
        args.out_dir, suffix=f"seed{config.simulation.seed}_{selection.label}")
    print(f"  出力先: {out_dir}")

    parent_drift = {}
    parent_metrics = locate(parent_run, "metrics.csv")
    if parent_metrics is not None:
        import csv as _csv
        with open(parent_metrics, newline="", encoding="utf-8") as handle:
            rows = list(_csv.DictReader(handle))[-3:]
        parent_drift = {"note": "sham 無しなので、回復幅がこのドリフト幅と同オーダーなら結論を出さない",
                        "tail_rows": rows}

    manifest = {
        "parent_run": str(parent_run.resolve()),
        "restored_from_hour": from_hour,
        "cut_specs": list(args.cut),
        "cut_combine": args.cut_combine,
        "label": selection.label,
        "description": selection.description,
        "num_synapses_before": int(coo.row.size),
        "num_synapses_cut": num_cut,
        "num_synapses_after": int(coo.row.size - num_cut),
        "selector_detail": selection.detail,
        "cut_profile": cut_summary,
        "cross_check": cross_check,
        "verification": verification,
        "avalanche_smax": smax,
        "hub_z": args.hub_z,
        "preserve_fan_in_scale": True,
        "parent_drift": parent_drift,
        "timeline_ms": {
            "settle": args.settle_ms,
            "baseline_window": args.baseline_window_ms or args.probe_window_ms,
            "probe_window": args.probe_window_ms,
            "probe_interval": args.probe_interval_hours * HOUR_MS,
            "recovery_total": args.recovery_hours * HOUR_MS,
        },
    }
    runio.save_manifest(out_dir, manifest)
    runio.save_cut(out_dir, per_synapse)
    runio.write_rows_csv(comparison, out_dir / runio.CUT_PROFILE_NAME)
    ConfigManager().save_config(config, save_dir=out_dir)
    layout.save_axes(out_dir / AXES_NAME)

    if args.dry_run:
        print(json.dumps({k: manifest[k] for k in
                          ("label", "num_synapses_before", "num_synapses_cut",
                           "num_synapses_after", "cross_check", "verification")},
                         ensure_ascii=False, indent=2))
        print(f"--dry-run: GeNN は実行していません。{out_dir}")
        return

    dt = float(config.simulation.dt)
    config.task.record_buffer_ms = args.record_buffer_ms
    chunk_steps = max(1, int(args.record_buffer_ms / dt))
    baseline_window = args.baseline_window_ms or args.probe_window_ms
    stem = Path(require(parent_run, CONFIG_NAME)).parent.name
    seed_tag = f"seed{config.simulation.seed}"

    # ---------------- Phase 1: intact ----------------
    print("\n=== Phase 1: 損傷なしで復元し、切断直前のベースラインを測る ===")
    b1, model1, layout1 = build_network(config, f"lesion_intact_{seed_tag}",
                                        args.genn_code_dir, restored_weights=restored)
    if axes_path is not None:
        layout1.load_axes_file(axes_path)
    sim1 = GeNNSimulator(model1, config, b1)
    sim1.setup(backup_initial_states=False)   # 復元前の重みを initial_states に残さない
    if args.settle_ms > 0:
        run_steps(sim1, int(round(args.settle_ms / dt)), chunk_steps, keep_spikes=False)

    rows = []
    # baseline は**切断より前**に測った値なので負の時刻に置く。0.0 にすると post の
    # 最初の probe と重なり、「切断の瞬間」が図でも CSV でも判別できなくなる。
    baseline_hours_ms = -baseline_window
    probes = [{"index": 0, "phase": runio.PHASE_BASELINE,
               "hours_since_cut": baseline_hours_ms / HOUR_MS,
               "window_ms": baseline_window}]
    rows.append(probe(sim1, b1, layout1, config, out_dir, 0, runio.PHASE_BASELINE,
                      baseline_hours_ms, baseline_window, chunk_steps, smax, args))
    w_pre = sim1.pull_synapse_coo("w")
    np.savez_compressed(out_dir / runio.PRE_WEIGHTS_NAME,
                        row=w_pre["row"], col=w_pre["col"], data=w_pre["data"])
    print(f"  切断直前の重みを保存: {w_pre['data'].size} 本")
    del sim1, model1, b1

    # ---------------- Phase 2: lesioned ----------------
    print(f"\n=== Phase 2: {num_cut} 本を構造的に除去して回復を追う ===")
    coo2_source = align_saved_to_coo(w_pre["row"], w_pre["col"], w_pre["data"], coo)
    b2, model2, layout2 = build_network(
        config, f"lesion_cut_{seed_tag}_{selection.label}"[:80], args.genn_code_dir,
        restored_weights=coo2_source, cut_mask=selection.cut, preserve_fan_in=True)
    if axes_path is not None:
        layout2.load_axes_file(axes_path)
    sim2 = GeNNSimulator(model2, config, b2)
    sim2.setup(backup_initial_states=False)

    check = sim2.pull_synapse_coo("w")
    expected = coo2_source[~selection.cut]
    aligned = align_saved_to_coo(check["row"], check["col"], check["data"], b2.global_coo())
    if not np.allclose(aligned, expected, rtol=0, atol=1e-6):
        raise RuntimeError("Phase 2 の復元重みが Phase 1 の値と一致しません (復元経路の破損)。")
    print("  Phase 1 の重みが残存シナプスへ正しく入ったことを確認しました")

    connectivity2 = sim2.synapse_connectivity_coo()
    np.savez_compressed(out_dir / CONNECTIVITY_NAME, row=connectivity2["row"],
                        col=connectivity2["col"], shape=connectivity2["shape"])

    if not args.no_structure_figures:
        try:
            # **切断後の幾何を渡すこと。** builder のコネクタが持つ幾何は
            # replace_global_coo() の影響を受けないので、既定のままだと axon_network.png に
            # 切ったはずの結合まで描かれる。
            post_geometry = (subset_geometry(geometry, ~selection.cut)
                             if geometry is not None else None)
            visualize_structure(b2, config, out_dir / "structure_post", seed=0,
                                order_axes=resolve_order_axes(layout2),
                                geometry=post_geometry)
        except Exception as error:
            print(f"  Warning: 構造図の生成に失敗しました: {error}")

    if args.settle_ms > 0:
        run_steps(sim2, int(round(args.settle_ms / dt)), chunk_steps, keep_spikes=False)

    num_probes = max(1, int(round(args.recovery_hours / args.probe_interval_hours)) + 1)
    elapsed_ms = 0.0
    for k in range(num_probes):
        target_ms = k * args.probe_interval_hours * HOUR_MS
        gap_ms = target_ms - elapsed_ms
        if gap_ms > 0:
            run_steps(sim2, int(round(gap_ms / dt)), chunk_steps, keep_spikes=False)
            elapsed_ms += gap_ms
        index = k + 1
        print(f"  probe {index}/{num_probes}  t = {elapsed_ms / HOUR_MS:.2f} h")
        rows.append(probe(sim2, b2, layout2, config, out_dir, index, runio.PHASE_POST,
                          elapsed_ms, args.probe_window_ms, chunk_steps, smax, args))
        elapsed_ms += args.probe_window_ms
        probes.append({"index": index, "phase": runio.PHASE_POST,
                       "hours_since_cut": rows[-1]["hours_since_cut"],
                       "window_ms": args.probe_window_ms})

    runio.write_rows_csv(probes, out_dir / runio.PROBES_NAME)
    runio.write_rows_csv(lesion_metrics.delta_from_baseline(rows, rows[0]),
                         out_dir / runio.METRICS_NAME)

    # develop.py の figure2c 左列にあたる重みの遷移。切断前後を同じ図に載せるため、
    # baseline は生き残ったシナプスへ引き当ててから描く (figures.py が面倒を見る)。
    print("\nGenerating figures...")
    for draw in (lesion_figures.plot_weight_trajectories,
                 lesion_figures.plot_weight_distribution_shift,
                 lesion_figures.plot_firing_rate_scatter):
        try:
            draw(out_dir, layout2)
        except Exception as error:
            print(f"  Warning: {draw.__name__} に失敗しました: {error}")

    print(f"\n記録を保存しました: {out_dir}")
    try:
        organize_output(out_dir)
    except Exception as error:
        print(f"  Warning: データ整理に失敗しました: {error}")


if __name__ == "__main__":
    main()
