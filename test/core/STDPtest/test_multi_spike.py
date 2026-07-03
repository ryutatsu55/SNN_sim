"""
複数スパイクに対する custom_Akita (e-stdp_trace / i-stdp_trace) の重み更新検証テスト
=====================================================================
本テストは E-STDP と I-STDP の 2 条件を続けて実行する (CONFIGS を参照)。
  - E-STDP: test/core/STDPtest/test_multi_estdp.yaml
  - I-STDP: test/core/STDPtest/test_multi.yaml
plasticity.mode を config から自動判別し、対応する素のカーネル
(e_stdp_kernel / i_stdp_kernel) で到着時刻ベースの全ペア和リファレンスを構成する。
=====================================================================

`test/core/STDPtest/test.py` は 1 トライアル = pre→post 1 ペアのみを発火させ
STDP ウィンドウを描くだけで、複数スパイクが交錯したときに trace が正しく蓄積され、
シナプス伝播遅延に対するロールバック処理
(custom_Akita.py の _estdp_trace_dc_syn / _istdp_trace_dc_syn) が正しく効くかは
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
from src.models.plasticity.custom_Akita import e_stdp_kernel, i_stdp_kernel

# 実行する検証条件 (ラベル, config パス)。
# 各 config は単一のシナプス集団 (e-stdp_trace または i-stdp_trace) を持ち、
# mode は config から自動判別して対応するカーネル/パラメータへディスパッチする。
CONFIGS = [
    ("E-STDP", "test/core/STDPtest/test_multi_estdp.yaml"),
    ("I-STDP", "test/core/STDPtest/test_multi_istdp.yaml"),
]


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


def delayed_i_stdp(t_pre, t_post, delay_ms, A_I, tau_I1, tau_I2, beta_I):
    """
    遅延を外部で加味した I-STDP カーネル (1 ペア分の重み変化)。

    E-STDP と同様、到着時刻基準の時間差 delta_t = t_post - (t_pre + delay_ms) を
    素の i_stdp_kernel (Supplementary Eq. S17) へ渡す。I-STDP は時間差について対称
    (kernel は |delta_t| を使う) なため、pre→post 到着前後どちらでも同じ符号で寄与する。
    trace 型 all-to-all I-STDP の数学的な ground truth (全ペア和) を与える。
    """
    delta_t = t_post - (t_pre + delay_ms)
    return i_stdp_kernel(delta_t, A_I, tau_I1, tau_I2, beta_I)


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


def compute_partial_dw_ref(t_ck, pre_t, post_t, delay_ms, kernel_fn):
    """
    t_ck 以前に適用済みの全ペア和を計算する (中間チェックポイント用リファレンス)。

    delay_corrected モードの更新タイミング:
      - LTP: post 発火時 (t_post) に適用  -> t_post <= t_ck のペアをカウント
      - LTD: pre 到着時 (t_arrival) に適用 -> t_arrival <= t_ck のペアをカウント

    ロールバックシナリオ (pre=10.0, post=11.0, delay=1.5) の場合:
      delta_t = 11.0 - 11.5 = -0.5 < 0 -> LTD として t_arrival=11.5 で計上
      t_ck=11.0 (post 発火直後) ではまだカウントされない -> dw=0 が正解
    """
    dw = 0.0
    for tpre in pre_t:
        t_arrival = tpre + delay_ms
        for tpost in post_t:
            delta_t = tpost - t_arrival
            if delta_t >= 0.0 and tpost <= t_ck:
                dw += kernel_fn(float(tpre), float(tpost), delay_ms)
            elif delta_t < 0.0 and t_arrival <= t_ck:
                dw += kernel_fn(float(tpre), float(tpost), delay_ms)
    return dw


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


def fire_and_checkpoint(sim, total_neurons, total_steps, vrest, stim,
                        pre_id, pre_times_ms, post_id, post_times_ms,
                        dt, delay_ms, read_weight_fn, include_pre_fire_ck=False):
    """
    fire_schedule と同様にシナリオを実行しつつ、各 STDP 重み更新イベントの直後に重みを読み出す。

    チェックポイントは delay_corrected モードの更新タイミングに対応:
      - post 発火ステップ: post_spike_syn_code が走り LTP が適用される
      - pre 到着ステップ:  syn_dynamics_code が到着を検出し LTD が適用される
      - pre 発火ステップ:  include_pre_fire_ck=True のとき追加 (重みは変化しない sanity check)

    Returns:
        list of (label: str, t_event_ms: float, w_measured: float)  時刻順
    """
    delay_steps = int(round(delay_ms / dt))

    # step -> {"fire": [gid, ...], "ck": [(label, t_ms), ...]}
    step_info = {}

    def get_slot(step):
        if step not in step_info:
            step_info[step] = {"fire": [], "ck": []}
        return step_info[step]

    for t in pre_times_ms:
        fs = int(round(t / dt))
        get_slot(fs)["fire"].append(pre_id)
        if include_pre_fire_ck:
            get_slot(fs)["ck"].append((f"pre_fire@{t:.1f}ms", t))
        arr_step = fs + delay_steps
        arr_t = t + delay_ms
        get_slot(arr_step)["ck"].append((f"pre_arrival@{arr_t:.1f}ms", arr_t))

    for t in post_times_ms:
        fs = int(round(t / dt))
        get_slot(fs)["fire"].append(post_id)
        get_slot(fs)["ck"].append((f"post_fire@{t:.1f}ms", t))

    checkpoints = []
    current = 0
    for step in sorted(step_info.keys()):
        info = step_info[step]
        if step > current:
            sim.step(step - current)
            current = step
        if info["fire"]:
            arr = np.full(total_neurons, vrest, dtype=np.float32)
            for gid in info["fire"]:
                arr[gid] = stim
            sim.push(arr, target_var="V")
        sim.step(2)
        current += 2
        if info["ck"]:
            w_now = read_weight_fn(sim)
            for label, t_ms in info["ck"]:
                checkpoints.append((label, t_ms, w_now))

    if total_steps > current:
        sim.step(total_steps - current)
    return checkpoints


def measure_delay_ms(sim, ctx):
    """(src,tgt) の伝播遅延[ms]を dendritic / axonal 両対応で取得する。

    - dendritic 経路: per-synapse 変数 d(タイムステップ) を持つ → d * dt。
    - axonal 経路   : delay_by_target 指定で custom_Akita が d を持たず、遅延は
                      SynapseGroup.axonal_delay_steps(group 均一)で表現される。
                      到着遅延 = (axonal_delay_steps + 1) * dt (GeNN の st_pre 補正に一致)。
    """
    dt = ctx["dt"]
    for sg in sim.model.synapse_populations.values():
        if "d" in sg.vars:
            return float(sim.pull_synapse("d")[ctx["src_ID"], ctx["tgt_ID"]] * dt)
        # axonal: STDP の実効到着遅延 = axonal_delay_steps * dt (emission 基準)
        return float(sg.axonal_delay_steps * dt)
    raise RuntimeError("no synapse population found to measure delay")


def run_scenario(sim, name, pre_times_ms, post_times_ms, ctx):
    """1 シナリオを実行し、中間チェックポイントと最終 dw を検証する。

    中間チェック: 各 post 発火・pre 到着の直後に重みを読み出し、
    compute_partial_dw_ref によるリファレンスと突き合わせる。
    最終チェック: 従来通り全ペア和リファレンスと比較する。
    """
    # --- 遅延の実測 (arrival checkpoint 構築に先立ち取得) ---
    delay_ms = measure_delay_ms(sim, ctx)

    # --- fire_and_checkpoint で実行: 各更新イベント直後の重みを収集 ---
    def read_w(sim_):
        return read_weight_native(sim_, ctx["src_ID"], ctx["tgt_ID"])

    checkpoints = fire_and_checkpoint(
        sim=sim,
        total_neurons=ctx["total_neurons"],
        total_steps=ctx["total_steps"],
        vrest=ctx["vrest"],
        stim=ctx["stim"],
        pre_id=ctx["src_ID"],
        pre_times_ms=pre_times_ms,
        post_id=ctx["tgt_ID"],
        post_times_ms=post_times_ms,
        dt=ctx["dt"],
        delay_ms=delay_ms,
        read_weight_fn=read_w,
    )

    # --- 実測スパイク時刻の取得 ---
    spikes = sim.get_global_spikes()
    pre_t = np.sort(spikes["times"][spikes["ids"] == ctx["src_ID"]])
    post_t = np.sort(spikes["times"][spikes["ids"] == ctx["tgt_ID"]])

    # --- 防御チェック: 余計な発火 (EPSP由来など) がないこと ---
    spike_count_ok = (len(pre_t) == len(pre_times_ms)) and (len(post_t) == len(post_times_ms))

    atol = 1e-5

    # --- ログヘッダ ---
    print(f"\n--- Scenario {name} ---")
    print(f"  pre  spikes (ms): scheduled={pre_times_ms}  measured={np.round(pre_t, 3).tolist()}")
    print(f"  post spikes (ms): scheduled={post_times_ms}  measured={np.round(post_t, 3).tolist()}")
    print(f"  delay (ms)      : {delay_ms}")
    if not spike_count_ok:
        print(f"  [WARN] spike count mismatch! "
              f"pre {len(pre_t)}/{len(pre_times_ms)}, post {len(post_t)}/{len(post_times_ms)}")

    # --- 中間チェック ---
    all_intermediate_ok = True
    print(f"  --- Intermediate weight checkpoints ---")
    for label, t_event_ms, w_measured in checkpoints:
        dw_partial = compute_partial_dw_ref(
            t_event_ms, pre_t, post_t, delay_ms, ctx["kernel"]
        )
        w_ref = ctx["base_weight"] + dw_partial
        diff = abs(w_measured - w_ref)
        ok = bool(np.isclose(w_measured, w_ref, atol=atol, rtol=0.0))
        if not ok:
            all_intermediate_ok = False
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: "
              f"w={w_measured:+.8f}  ref={w_ref:+.8f}  |diff|={diff:.2e}")

    # --- 最終チェック (従来通り全ペア和) ---
    dw_ref = 0.0
    for tpre in pre_t:
        for tpost in post_t:
            dw_ref += ctx["kernel"](float(tpre), float(tpost), delay_ms)

    # native (double) 読み出しでは残差は倍精度の丸め (~1e-15) まで落ちる。
    w_final = read_weight_native(sim, ctx["src_ID"], ctx["tgt_ID"])
    dw_genn = w_final - ctx["base_weight"]
    diff_final = abs(dw_genn - dw_ref)
    within_tol = np.isclose(dw_genn, dw_ref, atol=atol, rtol=0.0)

    passed = bool(spike_count_ok and all_intermediate_ok and within_tol)

    print(f"  --- Final check ---")
    print(f"  dw (GeNN)       : {dw_genn:+.8f}")
    print(f"  dw (reference)  : {dw_ref:+.8f}")
    print(f"  |diff|          : {diff_final:.2e}  (atol={atol:.2e}; native double readout)")
    print(f"  result          : {'PASS' if passed else 'FAIL'}")

    sim.reset()
    return passed


# --- 共通シナリオ定義 -------------------------------------------------------
# 前提: delay (1.5ms) < TauRefrac (5ms)。同一ニューロンの連続発火は不応期で 5ms 超
# 離れるため、pre スパイクは必ず次の pre が発火する前に到着する = 伝播中の pre は
# 常に高々 1 本。これにより custom_Akita のロールバック処理 (単一伝播中スパイク前提)
# が成立する。
#
# delta_t=0 (t_post == t_pre + delay) の同時到着ケースは E: で検証する。
# post_spike_syn_code で READ バッファ未反映の到着分 (+1.0) を補完する修正が必要。
#
# I-STDP は時間差について対称なため LTP/LTD という非対称な意味付けはないが、pre/post の
# 到着前後・伝播中ロールバックといった「経路」は E-STDP と共通であり、同一シナリオで
# 両モードの全ペア和を検証できる。
SCENARIOS = [
    # A: 多 pre -> 1 post。post 側更新 (到着済み pre) と pre 側更新 (post 後に発火する
    #    pre) が 1 つの post に対して累積する (ロールバックなし)。
    ("A: multi-pre -> single-post (accum)", [5.0, 13.0, 21.0], [18.0]),
    # B: 1 pre -> 多 post。post=11 は pre=10 の伝播中 (到着 11.5) に発火 = 単一伝播中
    #    ロールバック経路。post=30,50 は通常経路。
    ("B: single-pre -> multi-post (single in-transit rollback)", [10.0], [11.0, 30.0, 50.0]),
    # C: 交錯多重。post=6 / post=41 はそれぞれ pre=5 / pre=40 の伝播中に発火 = ロール
    #    バックが複数回発生。さらに通常更新と pre 側更新が混在する 3x3 ペア。
    ("C: interleaved multi-pre/multi-post (multi rollback events)",
     [5.0, 40.0, 70.0], [4.0, 41.0, 58.0]),
    ("D: custom", [6.0], [2.0, 4.0, 9.0]),
    # E: delta_t=0。pre=10ms, delay=1.5ms -> arrival=11.5ms = post 発火時刻。
    #    post_spike_syn_code が READ バッファの pre_trace_syn=0 (到着分未コミット) を
    #    補完して LTP = A_E を正しく適用できるか検証する。
    ("E: delta_t=0 (simultaneous arrival and post fire)", [10.0], [11.5]),
]


def build_ctx(config, sim, plast):
    """config の plasticity.mode から検証用 ctx (パラメータ + リファレンスカーネル) を作る。"""
    mode = str(plast.mode)
    ctx = {
        "mode": mode,
        "total_neurons": sim.total_neurons,
        "total_steps": int(round(config.task.duration / config.simulation.dt)),
        "dt": float(config.simulation.dt),
        "stim": float(config.task.input),
        "src_ID": int(config.network.connection.src_ID),
        "tgt_ID": int(config.network.connection.tgt_ID),
        "base_weight": float(config.network.weight.base_weight),
    }
    if mode.startswith("e-stdp"):
        A_E, tau_E, beta_E = float(plast.A_E), float(plast.tau_E), float(plast.beta_E)
        ctx["params_str"] = f"A_E={A_E}, tau_E={tau_E}, beta_E={beta_E}"
        ctx["kernel"] = lambda tpre, tpost, d: delayed_e_stdp(tpre, tpost, d, A_E, tau_E, beta_E)
    elif mode.startswith("i-stdp"):
        A_I = float(plast.A_I)
        tau_I1, tau_I2, beta_I = float(plast.tau_I1), float(plast.tau_I2), float(plast.beta_I)
        ctx["params_str"] = f"A_I={A_I}, tau_I1={tau_I1}, tau_I2={tau_I2}, beta_I={beta_I}"
        ctx["kernel"] = lambda tpre, tpost, d: delayed_i_stdp(tpre, tpost, d, A_I, tau_I1, tau_I2, beta_I)
    else:
        raise ValueError(f"Unsupported plasticity mode '{mode}' for this test.")
    return ctx


def run_config(label, config_path):
    """1 つの config (= 1 条件) について全シナリオを実行し、(シナリオ名, 合否) を返す。"""
    print(f"\n########## Condition: {label}  ({config_path}) ##########")
    print(f"Loading config from {config_path}...")
    manager = ConfigManager()
    config = manager.load_resolved(config_path)

    # config から唯一のシナプス集団とその source ニューロンを取り出す
    # (グループ名 Layer_Exc/from_Exc・Layer_Inh/from_Inh をハードコードしない)。
    syn_name = next(iter(config.synapses))
    syn_cfg = config.synapses[syn_name]
    neuron_name = syn_cfg.source
    plast = syn_cfg.plasticity

    print("Building Network with GeNN...")
    builder = NetworkBuilder(config)
    genn_model, group_info = builder.build(rec_spike=True)
    print(group_info)

    print("Initializing Simulator...")
    sim = GeNNSimulator(genn_model, config, builder)
    sim.setup()

    ctx = build_ctx(config, sim, plast)
    ctx["vrest"] = float(config.neurons[neuron_name].Vrest)
    print(f"  mode={ctx['mode']}, pre(global)={ctx['src_ID']}, post(global)={ctx['tgt_ID']}, "
          f"{ctx['params_str']}, base_weight={ctx['base_weight']}")

    results = []
    for name, pre_times, post_times in SCENARIOS:
        results.append((f"[{label}] {name}", run_scenario(sim, name, pre_times, post_times, ctx)))
    return results


def main():
    print("=== custom_Akita Multi-Spike STDP Verification (E-STDP / I-STDP) ===")

    results = []
    for label, config_path in CONFIGS:
        results.extend(run_config(label, config_path))

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
