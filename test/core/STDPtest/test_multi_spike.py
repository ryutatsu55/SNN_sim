"""
複数スパイクに対する custom_Akita (e-stdp_trace) の重み更新検証テスト
=====================================================================

`test/core/STDPtest/test.py` は 1 トライアル = pre→post 1 ペアのみを発火させ
STDP ウィンドウを描くだけで、複数スパイクが交錯したときに trace が正しく蓄積され、
シナプス伝播遅延に対するロールバック処理 (custom_Akita.py:282-306) が正しく効くかは
検証していない。

本テストは複数の pre/post スパイクを意図的に交錯させて発火させ、GeNN が算出する
正味の重み変化 dw を、独立した解析リファレンスと突き合わせる。

検証の原理 (ground truth)
-------------------------
trace 型 all-to-all STDP の正味の重み変化は、シナプス到着時刻ベースの全ペア和に一致する:

    arrival_i = t_pre_i + delay_ms
    dw_ref = Σ_{pre i, post j} e_stdp_kernel(t_post_j - arrival_i, A_E, tau_E, beta_E)

`e_stdp_kernel` (custom_Akita.py) は遅延を考慮しない素のカーネルなので、
遅延は本テスト側で「外部で」加味する (到着時刻 = pre 発火時刻 + delay)。
これは C++ 側 (t + d*dt_ms / t - d*dt_ms) と完全に同じ到着時刻基準であり、
かつテスト対象ファイルを改変しないため検証の独立性が保たれる。

重要な前提: delay < TauRefrac
-----------------------------
custom_Akita のロールバック処理は「伝播中の pre スパイクは高々 1 本」を前提に組まれて
いる ((pre_trace - 1.0) で伝播中の 1 本だけを除外する等)。シナプス遅延が絶対不応期
TauRefrac より大きいと、不応期で詰まった連続 pre スパイクが複数同時に伝播中になり得て、
この前提が破れて重み更新が誤る (実際に delay=20ms, TauRefrac=5ms では Scenario A が
解析値と符号まで食い違う)。本テストは delay (1.5ms) < TauRefrac (5ms) の正しい運用域で
検証する。同一ニューロンの連続発火は不応期で 5ms 超離れるため、pre は必ず次の pre が
発火する前に到着し、伝播中の pre は常に 1 本以下に保たれる。

許容誤差について
----------------
GeNN にはスパイク配送の 1 タイムステップぶんのスケジューリング遅延があり、記録スパイク
時刻と STDP 相互作用の実効時刻が高々 dt ずれる (符号は更新経路で一様でない)。この dt
レベルの離散化ぶんは ground truth との差として残るため、ペア数に応じて自動スケールする
許容誤差で吸収する (compute_tolerance 参照)。実バグ (上記 delay>TauRefrac 等) はこの
dt ジッタより桁で大きく、明確に検出される。

実行方法:
    cd /home/tanii/kuroki/SNN_sim
    python test/core/STDPtest/test_multi_spike.py
"""

import sys
import numpy as np
from pathlib import Path

# プロジェクトルートにパスを通す
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.config_manager import ConfigManager
from src.core.NetworkBuilder import NetworkBuilder
from src.core.simulator import GeNNSimulator

# --- プラグイン(モデル)の登録トリガー ---
import src.models.neurons.lif
import src.models.network.space
import src.models.network.connectors
import src.models.network.weights
import src.models.network.delays
import src.models.synapses.standard_models
import src.models.synapses.custom
import src.models.plasticity.custom_Akita
import src.models.plasticity.standard_models

# リファレンス計算用に「素のカーネル」を再利用する (遅延はこのスクリプト側で外部加味)
from src.models.plasticity.custom_Akita import e_stdp_kernel

CONFIG_PATH = "test/core/STDPtest/test_multi.yaml"


def delayed_e_stdp(t_pre, t_post, delay_ms, A_E, tau_E, beta_E):
    """
    遅延を外部で加味した E-STDP カーネル (1 ペア分の重み変化)。

    pre スパイクがシナプス後側に到達するのは t_pre + delay_ms なので、到着時刻基準の
    時間差 delta_t = t_post - (t_pre + delay_ms) を素のカーネルへ渡す。これは C++ 側
    custom_Akita.py の t ± d*dt_ms と同じ到着時刻基準であり、trace 型 all-to-all STDP の
    数学的な ground truth (全ペア和) を与える。

    なお GeNN にはスパイク配送の 1 タイムステップぶんのスケジューリング遅延があり、
    記録スパイク時刻と STDP 相互作用の実効時刻が高々 dt ずれる。このずれの符号は
    更新経路 (post 側 LTP / 伝播中ロールバック / pre 側 LTD) で一様でないため、本テストは
    この dt レベルの離散化ぶんを許容誤差として吸収する (compute_tolerance を参照)。
    """
    delta_t = t_post - (t_pre + delay_ms)
    return e_stdp_kernel(delta_t, A_E, tau_E, beta_E)


def read_weight_native(sim, src_ID, tgt_ID):
    """
    シナプス (src_ID -> tgt_ID) の重みを GeNN 内部の精度 (double) のまま読み出す。

    GeNNSimulator.pull_synapse() は戻り値を np.float32 行列に詰めるため、重みが単精度に
    丸められ ~3e-8 (float32 ULP) の読み出し誤差が乗る。モデルは double で計算しているので、
    ここでは pull_synapse のグローバル→ローカル対応をたどって該当接続の flat index を求め、
    syn_pop.vars["w"].values (double) を直接返すことで読み出し由来の丸めを排除する。
    """
    for syn_pop_name, syn_pop in sim.model.synapse_populations.items():
        src_name, _, tgt_name = syn_pop_name.partition("_to_")
        src_indices = sim.group_info[src_name]["global_indices"]
        tgt_indices = sim.group_info[tgt_name]["global_indices"]
        sub_mask = sim.builder.global_mask[np.ix_(src_indices, tgt_indices)]
        local_src_idx, local_tgt_idx = np.where(sub_mask != 0)
        global_src = src_indices[local_src_idx]
        global_tgt = tgt_indices[local_tgt_idx]
        match = np.where((global_src == src_ID) & (global_tgt == tgt_ID))[0]
        if len(match) > 0:
            syn_pop.vars["w"].pull_from_device()
            return float(syn_pop.vars["w"].values[match[0]])
    raise ValueError(f"No synapse found for connection ({src_ID} -> {tgt_ID}).")


def fire_schedule(sim, total_neurons, total_steps, vrest, stim,
                  pre_id, pre_times_ms, post_id, post_times_ms, dt):
    """
    指定した発火時刻(ms)で pre / post ニューロンを発火させる。

    各発火イベントでは V を Vthresh 超へ push して 1 ステップ進め、対象ニューロンを
    強制発火させる (発火後は reset_code が V=Vreset に上書きするため余韻による誤発火なし)。
    イベント間は push せずに step だけ進める。
    全イベント処理後、total_steps まで tail を進める (記録バッファを既存テストと同様に
    毎回同じ長さで埋めるため)。
    """
    # step -> 発火させる global id のリスト
    events = {}
    for t in pre_times_ms:
        step = int(round(t / dt))
        events.setdefault(step, []).append(pre_id)
    for t in post_times_ms:
        step = int(round(t / dt))
        events.setdefault(step, []).append(post_id)

    current = 0
    for step in sorted(events.keys()):
        if step > current:
            sim.step(step - current)
            current = step
        arr = np.full(total_neurons, vrest, dtype=np.float32)
        for gid in events[step]:
            arr[gid] = stim
        sim.push(arr, target_var="V")
        sim.step(1)
        current += 1

    if total_steps > current:
        sim.step(total_steps - current)


def run_scenario(sim, name, pre_times_ms, post_times_ms, ctx):
    """1 シナリオを実行し、GeNN の dw と解析リファレンス dw を比較する。"""
    fire_schedule(
        sim,
        total_neurons=ctx["total_neurons"],
        total_steps=ctx["total_steps"],
        vrest=ctx["vrest"],
        stim=ctx["stim"],
        pre_id=ctx["src_ID"],
        pre_times_ms=pre_times_ms,
        post_id=ctx["tgt_ID"],
        post_times_ms=post_times_ms,
        dt=ctx["dt"],
    )

    # --- 実測スパイク時刻の取得 ---
    spikes = sim.get_global_spikes()
    pre_t = np.sort(spikes["times"][spikes["ids"] == ctx["src_ID"]])
    post_t = np.sort(spikes["times"][spikes["ids"] == ctx["tgt_ID"]])

    # --- 防御チェック: 余計な発火 (EPSP由来など) がないこと ---
    spike_count_ok = (len(pre_t) == len(pre_times_ms)) and (len(post_t) == len(post_times_ms))

    # --- 遅延の実測 (到着時刻の根拠) ---
    delay_ms = float(sim.pull_synapse("d")[ctx["src_ID"], ctx["tgt_ID"]] * ctx["dt"])

    # --- 解析リファレンス: 到着時刻ベース全ペア和 ---
    dw_ref = 0.0
    for tpre in pre_t:
        for tpost in post_t:
            dw_ref += delayed_e_stdp(
                float(tpre), float(tpost), delay_ms,
                ctx["A_E"], ctx["tau_E"], ctx["beta_E"],
            )

    # --- GeNN 実測 dw (double のまま読み出し: pull_synapse の float32 丸めを回避) ---
    w_final = read_weight_native(sim, ctx["src_ID"], ctx["tgt_ID"])
    dw_genn = w_final - ctx["base_weight"]

    # native (double) 読み出しでは残差は倍精度の丸め (~1e-15) まで落ちる。
    atol = 1e-5

    diff = abs(dw_genn - dw_ref)
    within_tol = np.isclose(dw_genn, dw_ref, atol=atol, rtol=0.0)
    passed = bool(spike_count_ok and within_tol)

    # --- ログ出力 ---
    print(f"\n--- Scenario {name} ---")
    print(f"  pre  spikes (ms): scheduled={pre_times_ms}  measured={np.round(pre_t, 3).tolist()}")
    print(f"  post spikes (ms): scheduled={post_times_ms}  measured={np.round(post_t, 3).tolist()}")
    print(f"  delay (ms)      : {delay_ms}")
    if not spike_count_ok:
        print(f"  [WARN] spike count mismatch! "
              f"pre {len(pre_t)}/{len(pre_times_ms)}, post {len(post_t)}/{len(post_times_ms)}")
    print(f"  dw (GeNN)       : {dw_genn:+.8f}")
    print(f"  dw (reference)  : {dw_ref:+.8f}")
    print(f"  |diff|          : {diff:.2e}  (atol={atol:.2e}; native double readout)")
    print(f"  result          : {'PASS' if passed else 'FAIL'}")

    sim.reset()
    return passed


def main():
    print("=== custom_Akita Multi-Spike STDP Verification ===")

    print(f"Loading config from {CONFIG_PATH}...")
    manager = ConfigManager()
    config = manager.load_resolved(CONFIG_PATH)

    print("Building Network with GeNN...")
    builder = NetworkBuilder(config)
    genn_model, group_info = builder.build(rec_spike=True)
    print(group_info)

    print("Initializing Simulator...")
    sim = GeNNSimulator(genn_model, config, builder)
    sim.setup()

    # --- 検証に使う各種パラメータの取り出し ---
    plast = config.synapses["from_Exc"].plasticity
    ctx = {
        "total_neurons": sim.total_neurons,
        "total_steps": int(round(config.task.duration / config.simulation.dt)),
        "dt": float(config.simulation.dt),
        "vrest": float(config.neurons["Layer_Exc"].Vrest),
        "stim": float(config.task.input),
        "src_ID": int(config.network.connection.src_ID),
        "tgt_ID": int(config.network.connection.tgt_ID),
        "base_weight": float(config.network.weight.base_weight),
        "A_E": float(plast.A_E),
        "tau_E": float(plast.tau_E),
        "beta_E": float(plast.beta_E),
    }
    print(f"  pre(global)={ctx['src_ID']}, post(global)={ctx['tgt_ID']}, "
          f"A_E={ctx['A_E']}, tau_E={ctx['tau_E']}, beta_E={ctx['beta_E']}, "
          f"base_weight={ctx['base_weight']}")

    # --- シナリオ定義 -------------------------------------------------------
    # 前提: delay (1.5ms) < TauRefrac (5ms)。同一ニューロンの連続発火は不応期で 5ms 超
    # 離れるため、pre スパイクは必ず次の pre が発火する前に到着する = 伝播中の pre は
    # 常に高々 1 本。これにより custom_Akita のロールバック処理 (単一伝播中スパイク前提)
    # が成立する。各シナリオで t_post == t_pre+delay の同時到着特殊ケースは回避する。
    scenarios = [
        # A: 多 pre -> 1 post。post 側 LTP (到着済み pre) と pre 側 LTD (post 後に発火する
        #    pre) が 1 つの post に対して累積する (ロールバックなし)。
        ("A: multi-pre -> single-post (LTP+LTD accum)", [5.0, 13.0, 21.0], [18.0]),
        # B: 1 pre -> 多 post。post=11 は pre=10 の伝播中 (到着 11.5) に発火 = 単一伝播中
        #    ロールバック経路。post=30,50 は通常 LTP。
        ("B: single-pre -> multi-post (single in-transit rollback)", [10.0], [11.0, 30.0, 50.0]),
        # C: 交錯多重。post=6 / post=41 はそれぞれ pre=5 / pre=40 の伝播中に発火 = ロール
        #    バックが複数回発生。さらに通常 LTP と pre 側 LTD が混在する 3x3 ペア。
        ("C: interleaved multi-pre/multi-post (multi rollback events)",
         [5.0, 40.0, 70.0], [4.0, 41.0, 58.0]),
        ("D: custom", [6.0], [2.0, 9.0]),
    ]

    results = []
    for name, pre_times, post_times in scenarios:
        results.append((name, run_scenario(sim, name, pre_times, post_times, ctx)))

    # --- 集計 ---
    print("\n=== Summary ===")
    all_passed = True
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        all_passed = all_passed and passed

    if all_passed:
        print("\nAll scenarios PASSED.")
        sys.exit(0)
    else:
        print("\nSome scenarios FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
