"""AkitaDai 先生モデルベースのカスタム STDP + STP 可塑性モデル。

`CustomAkitaModel` は GeNN の weight-update スニペットを 3 つの直交軸に応じて動的生成する:

    - 極性  : mode = e-stdp / i-stdp   ... 興奮性 / 抑制性 STDP (mode.startswith で判定)
    - pairing: nearest / trace          ... 最近接ペア / trace 型 all-to-all (明示フィールド)
    - axonal : delay_by_target 指定時のみ ... NetworkBuilder が axonal_delay_steps を渡す

    → 実装は pairing × axonal の 4 経路:
        nearest × 非axonal → _nearest_dc_syn (到着イベント駆動 per-syn)
        nearest × axonal   → _nearest_syn    (直接指数カーネル, addToPost)
        trace   × 非axonal → _trace_dc_syn   (delay_corrected per-syn)
        trace   × axonal   → _trace_legacy_syn(per-neuron trace, st_pre 到着補正)

加えて STP（短期可塑性: 資源回復 Eq. S11 / 消費 Eq. S12）を全経路共通で持つ。

このモジュール冒頭の関数群は上記ロジックの「数式の純 Python 参照実装」であり、
GeNN が生成する C++ と独立に同じ数式を計算する。`CustomAkitaModel` 本体はそのうち
`calculate_gmax_scale` のみ利用し、残りは検証テスト
(test/test_akita_soc.py, test/core/STDPtest/test_multi_spike.py) が import して
C++ 側の挙動と突き合わせるために使う。
"""

import pygenn
import numpy as np
from src.core.registry import PLASTICITY_MODELS
from .BASE_plasticity import BasePlasticityModel


# ======================================================================
# 数式の参照実装（検証テスト用 / Supplementary Eq. S11–S17）
#   GeNN の C++ スニペットと独立に同じ数式を Python で計算する。
#   `calculate_gmax_scale` のみ CustomAkitaModel 本体も利用する。
# ======================================================================
def recover_synaptic_resource(x: float, delta_t: float, tau_rec: float) -> float:
    """Supplementary Eq. S11 の閉形式。"""
    return 1.0 - ((1.0 - x) * np.exp(-delta_t / tau_rec))


def consume_synaptic_resource(x: float, utilization: float) -> tuple[float, float]:
    """Supplementary Eq. S12 に従い資源を消費する。"""
    released = utilization * x
    return x - released, released


def calculate_gmax_scale(num_synapses: int, num_post: int, normalize_by_fan_in: bool) -> float:
    """g_maxを入力総量として扱うための平均fan-in正規化係数。"""
    if not normalize_by_fan_in:
        return 1.0
    if num_synapses <= 0 or num_post <= 0:
        return 1.0
    fan_in = num_synapses / num_post
    if fan_in <= 0.0:
        return 1.0
    return 1.0 / fan_in


def e_stdp_kernel(delta_t: float, a_e: float, tau_e: float, beta_e: float) -> float:
    """Supplementary Eq. S16。"""
    if delta_t >= 0.0:
        return a_e * np.exp(-delta_t / tau_e)
    return -a_e * beta_e * np.exp(delta_t / tau_e)


def i_stdp_kernel(delta_t: float, a_i: float, tau_i1: float, tau_i2: float, beta_i: float) -> float:
    """Supplementary Eq. S17。"""
    coeff = a_i / (1.0 - ((tau_i1 / tau_i2) * beta_i))
    abs_dt = abs(delta_t)
    return coeff * (
        np.exp(-abs_dt / tau_i1) - ((tau_i1 / tau_i2) * beta_i * np.exp(-abs_dt / tau_i2))
    )


def decay_trace(trace: float, elapsed: float, tau: float) -> float:
    """指数減衰traceをelapsedだけ進める。"""
    if elapsed <= 0.0:
        return trace
    return trace * np.exp(-elapsed / tau)


def e_trace_pre_delta(post_trace: float, a_e: float, beta_e: float) -> float:
    """E-STDP trace型でpre spike時に適用する重み変化。"""
    return -a_e * beta_e * post_trace


def e_trace_post_delta(pre_trace: float, a_e: float) -> float:
    """E-STDP trace型でpost spike時に適用する重み変化。"""
    return a_e * pre_trace


def i_trace_delta(trace1: float, trace2: float, a_i: float, tau_i1: float, tau_i2: float, beta_i: float) -> float:
    """I-STDP trace型で相手側traceから重み変化を計算する。"""
    c_i_beta = (tau_i1 / tau_i2) * beta_i
    c_i = a_i / (1.0 - c_i_beta)
    return c_i * (trace1 - (c_i_beta * trace2))


@PLASTICITY_MODELS.register("custom_Akita")
class CustomAkitaModel(BasePlasticityModel):
    """
    AkitaDai先生のモデルをベースにしたカスタムSTDP+STPモデル。
    """
    # PyGeNNにおける「同じクラス名の二重登録エラー」を防ぐための管理辞書(同一クラスのインスタンス間をまたぐ変数)
    _snippet_cache = {}

    def __init__(self, config, dt, weight, delay, num_pre, num_post, axonal_delay_steps=None):
        super().__init__(config, dt, weight, delay, num_pre, num_post, axonal_delay_steps)
        # 極性: mode プレフィックスで判定 (従来どおり)。mode は極性 + param ブロックキーを兼ねる。
        self.mode = self.config.mode
        if not (self.mode.startswith("e-stdp") or self.mode.startswith("i-stdp")):
            raise ValueError(f"Unsupported mode '{self.mode}' for CustomAkitaModel.")
        self._is_e_stdp = self.mode.startswith("e-stdp")
        # pairing: nearest / trace を明示フィールドで指定する。
        self.pairing = getattr(self.config, "pairing", None)
        if self.pairing not in ("nearest", "trace"):
            raise ValueError(
                f"CustomAkitaModel requires pairing='nearest' or 'trace' (got {self.pairing!r})."
            )
        self.trace_mode = self.pairing == "trace"
        # 軸索遅延経路: NetworkBuilder が delay_by_target(=集団内均一遅延)を検出したとき有効化される。
        # GeNN が axonal_delay_steps 分だけ遅延スロットからスパイクを読み、pre_spike_syn_code を
        # 到着時刻にイベント駆動で実行し、st_pre も到着時刻へ補正する。よって毎ステップの
        # syn_dynamics_code は不要になり、STDP は到着時刻基準(=遅延補正済み)の数式で
        # pre/post_spike_syn_code に直接書ける。電流は addToPostDelay ではなく addToPost で即時投与する。
        self._axonal = self.axonal_delay_steps is not None
        # d はタイムステップ単位(uint8)。GeNN の t は ms 単位なので、STDP の時間演算では
        # d を ms に換算する (d * dt_ms)。※ addToPostDelay の遅延引数はタイムステップ単位のため d のまま使う。
        self._dt_ms = float(self.dt)
        self._gmax_scale = calculate_gmax_scale(
            num_synapses=len(self.weight),
            num_post=self.num_post,
            normalize_by_fan_in=bool(getattr(self.config, "normalize_gmax_by_fan_in", False)),
        )

        self._params, self._vars, self._pre_vars, self._post_vars = self._prepare_genn_data()

        # スニペット C++ は param 値を焼き込まず init_weight_update の params 経由で受け取るため、
        # キャッシュキーは (極性, pairing, axonal) のみで十分 (param 変種をまたいで共有可)。
        cache_key = ("e" if self._is_e_stdp else "i", self.pairing, self._axonal)
        if cache_key not in CustomAkitaModel._snippet_cache:
            # 未登録の場合のみ生成
            CustomAkitaModel._snippet_cache[cache_key] = self._create_snippet()
        self._custom_snippet_obj = CustomAkitaModel._snippet_cache[cache_key]

    # ==========================================
    # 1. パラメータと変数の準備
    # ==========================================
    def _prepare_genn_data(self):
        # 変数設定 (モード共通)。axonal 経路では per-synapse 遅延 d は不要(集団均一の軸索遅延を使う)。
        vars_dict = {
            "w": self.weight.astype('float32'),
        }
        if not self._axonal:
            vars_dict["d"] = self.delay.astype('uint8')
        pre_vars_dict = {
            "x": np.ones(self.num_pre, dtype='float32'),
            "x_release": np.zeros(self.num_pre, dtype='float32'),
        }
        post_vars_dict = {}

        if self.trace_mode:
            if self._is_e_stdp:
                # 非axonal(delay_corrected): pre_trace を per-synapse vars に置く（到着時刻に更新するため）
                # axonal: per-neuron pre_vars に置く (st_pre が到着時刻へ自動補正される)
                if self._axonal:
                    pre_vars_dict["pre_trace"] = np.zeros(self.num_pre, dtype='float64')
                else:
                    vars_dict["pre_trace_syn"] = np.zeros(len(self.weight), dtype='float64')
                post_vars_dict.update({
                    "post_trace": np.zeros(self.num_post, dtype='float64'),
                })
            else:
                if self._axonal:
                    pre_vars_dict["pre_trace1"] = np.zeros(self.num_pre, dtype='float64')
                    pre_vars_dict["pre_trace2"] = np.zeros(self.num_pre, dtype='float64')
                else:
                    vars_dict["pre_trace_syn1"] = np.zeros(len(self.weight), dtype='float64')
                    vars_dict["pre_trace_syn2"] = np.zeros(len(self.weight), dtype='float64')
                post_vars_dict.update({
                    "post_trace1": np.zeros(self.num_post, dtype='float64'),
                    "post_trace2": np.zeros(self.num_post, dtype='float64'),
                })
        elif self._axonal:
            # nearest × axonal: 直接カーネル(_nearest_syn)を使うため trace 変数は一切不要。
            pass
        else:
            # nearest モード: per-synapse trace + post trace（= 1.0 上書きで nearest-neighbor を実現）
            if self._is_e_stdp:
                vars_dict["pre_trace_syn"] = np.zeros(len(self.weight), dtype='float64')
                post_vars_dict["post_trace"] = np.zeros(self.num_post, dtype='float64')
            else:
                vars_dict["pre_trace_syn1"] = np.zeros(len(self.weight), dtype='float64')
                vars_dict["pre_trace_syn2"] = np.zeros(len(self.weight), dtype='float64')
                post_vars_dict["post_trace1"] = np.zeros(self.num_post, dtype='float64')
                post_vars_dict["post_trace2"] = np.zeros(self.num_post, dtype='float64')

        # モードに応じたパラメータ取得メソッドへディスパッチ
        if self._is_e_stdp:
            params = self._get_e_stdp_params()
        else:
            params = self._get_i_stdp_params()

        return params, vars_dict, pre_vars_dict, post_vars_dict

    def _get_e_stdp_params(self):
        tau_E = float(self.config.tau_E)
        return {
            "tau_rec": float(self.config.tau_rec),
            "U": float(self.config.U),
            "g_max": float(self.config.g_max),
            "g_scale": float(self._gmax_scale),
            "A_E": float(self.config.A_E),
            "tau_E": tau_E,
            "beta_E": float(self.config.beta_E),
            "decay_E": float(np.exp(-self._dt_ms / tau_E)),
            "wMin": float(self.config.Wmin),
            "wMax": float(self.config.Wmax)
        }

    def _get_i_stdp_params(self):
        tau_I1 = float(self.config.tau_I1)
        tau_I2 = float(self.config.tau_I2)
        beta_I = float(self.config.beta_I)
        A_I = float(self.config.A_I)

        C_I_beta = (tau_I1 / tau_I2) * beta_I
        C_I = A_I / (1.0 - C_I_beta)

        return {
            "tau_rec": float(self.config.tau_rec),
            "U": float(self.config.U),
            "g_max": float(self.config.g_max),
            "g_scale": float(self._gmax_scale),
            "tau_I1": tau_I1,
            "tau_I2": tau_I2,
            "C_I": C_I,
            "C_I_beta": C_I_beta,
            "decay_I1": float(np.exp(-self._dt_ms / tau_I1)),
            "decay_I2": float(np.exp(-self._dt_ms / tau_I2)),
            "wMin": float(self.config.Wmin),
            "wMax": float(self.config.Wmax)
        }

    # ==========================================
    # 2. GeNN スニペット生成
    # ==========================================
    def _create_snippet(self):
        (pre_code, post_code, pre_syn_code, post_syn_code,
         syn_dyn_code, pre_arrival_syn_code) = self._build_cpp_codes()

        # クラス名は極性 × pairing × axonal で一意 (param 変種はスニペット構造に影響しないため含めない)。
        safe_mode = "e_stdp" if self._is_e_stdp else "i_stdp"
        if not self.trace_mode:
            alg_suffix = "nrst"
        elif self._axonal:
            # axonal: per-neuron trace レイアウト (st_pre 到着補正で遅延補正済み)
            alg_suffix = "leg"
        else:
            alg_suffix = "dc"
        if self._axonal:
            # addToPostDelay/addToPost や d の有無が異なるため、別クラス名にして二重登録を防ぐ
            alg_suffix = "axo_" + alg_suffix

        # per-synapse vars: w は共通。d は axonal 以外のみ。delay_corrected/nearest(非axonal)では
        # pre_trace_syn も per-synapse。
        var_defs = [("w", "scalar")]
        if not self._axonal:
            var_defs.append(("d", "uint8_t"))
        pre_var_defs = [("x", "scalar"), ("x_release", "scalar")]
        post_var_defs = []

        if self.trace_mode:
            if self._is_e_stdp:
                if self._axonal:
                    pre_var_defs.append(("pre_trace", "double", pygenn.VarAccess.READ_WRITE))
                else:
                    var_defs.append(("pre_trace_syn", "double"))
                post_var_defs.append(("post_trace", "double", pygenn.VarAccess.READ_WRITE))
            else:
                if self._axonal:
                    pre_var_defs.append(("pre_trace1", "double", pygenn.VarAccess.READ_WRITE))
                    pre_var_defs.append(("pre_trace2", "double", pygenn.VarAccess.READ_WRITE))
                else:
                    var_defs.append(("pre_trace_syn1", "double"))
                    var_defs.append(("pre_trace_syn2", "double"))
                post_var_defs.append(("post_trace1", "double", pygenn.VarAccess.READ_WRITE))
                post_var_defs.append(("post_trace2", "double", pygenn.VarAccess.READ_WRITE))
        elif self._axonal:
            # nearest × axonal: 直接カーネルで trace 変数不要
            pass
        else:
            # nearest モード: per-synapse trace + post trace
            if self._is_e_stdp:
                var_defs.append(("pre_trace_syn", "double"))
                post_var_defs.append(("post_trace", "double", pygenn.VarAccess.READ_WRITE))
            else:
                var_defs.append(("pre_trace_syn1", "double"))
                var_defs.append(("pre_trace_syn2", "double"))
                post_var_defs.append(("post_trace1", "double", pygenn.VarAccess.READ_WRITE))
                post_var_defs.append(("post_trace2", "double", pygenn.VarAccess.READ_WRITE))

        snippet_kwargs = dict(
            class_name=f"custom_Akita_{safe_mode}_{alg_suffix}_gscaled",
            params=list(self._params.keys()),
            vars=var_defs,
            pre_vars=pre_var_defs,
            post_vars=post_var_defs,
            pre_spike_code=pre_code,
            post_spike_code=post_code,
            pre_spike_syn_code=pre_syn_code,
            post_spike_syn_code=post_syn_code,
            synapse_dynamics_code=syn_dyn_code,
        )
        # delay_corrected / nearest_dc の非軸索パスは per-synapse 到着イベント駆動。
        # GeNN fork の pre_arrival_syn_code (到着時刻に実行) + arrival_delay_var="d" を使う。
        if pre_arrival_syn_code is not None:
            snippet_kwargs["pre_arrival_syn_code"] = pre_arrival_syn_code
            snippet_kwargs["arrival_delay_var"] = "d"

        snippet = pygenn.create_weight_update_model(**snippet_kwargs)

        return snippet

    # ==========================================
    # 3. C++ ロジック生成
    # ==========================================
    def _build_cpp_codes(self):
        """C++のロジックコードを生成する。

        生成物は 6 つの C++ 文字列:
          - pre_spike_code / post_spike_code        : ニューロン発火時の状態更新 (STP 資源・trace)
          - pre_spike_syn_code / post_spike_syn_code: シナプス上での伝播・重み更新
          - syn_dyn_code                            : 毎ステップ実行 (現在は未使用 = None)
          - pre_arrival_syn_code                    : per-synapse 到着時刻にイベント駆動実行
                                                      (delay_corrected / nearest_dc の非軸索パス)
        mode (e/i-stdp) × trace/nearest × legacy/delay_corrected で分岐し、
        実際の生成は下記の per-mode ヘルパへ委譲する。
        """
        pre_spike_code = self._pre_spike_code()
        post_spike_code = self._post_spike_code()
        (pre_spike_syn_code, post_spike_syn_code,
         syn_dyn_code, pre_arrival_syn_code) = self._build_syn_codes()
        return (pre_spike_code, post_spike_code, pre_spike_syn_code,
                post_spike_syn_code, syn_dyn_code, pre_arrival_syn_code)

    def _pre_spike_code(self):
        """プレニューロン発火時: STP 資源更新 + (trace × axonal のみ) pre trace 更新。
        非 axonal の trace(delay_corrected) では pre_trace_syn を到着イベント(pre_arrival_syn_code)で
        更新するため、ここでは trace 更新を行わない。
        """
        pre_spike_code = f"""
            const scalar dt_pre = t - st_pre;
            x = 1.0 - ((1.0 - x) * exp(-dt_pre / tau_rec));
            x_release = U * x;
            x -= x_release;
        """

        if self._axonal and self.trace_mode:
            # trace × axonal は per-neuron pre_trace を発火時に更新する
            if self._is_e_stdp:
                pre_spike_code += f"""
                    pre_trace *= exp(-(t - st_pre) / tau_E);
                    pre_trace += 1.0;
                """
            else:
                pre_spike_code += f"""
                    const scalar dt_pre_trace = t - st_pre;
                    pre_trace1 *= exp(-dt_pre_trace / tau_I1);
                    pre_trace2 *= exp(-dt_pre_trace / tau_I2);
                    pre_trace1 += 1.0;
                    pre_trace2 += 1.0;
                """
        return pre_spike_code

    def _post_spike_code(self):
        """ポストニューロン発火時: post trace 更新。
        trace_mode: decay + += 1.0 (all-to-all 蓄積)
        nearest:    = 1.0 上書き (直近スパイクのみ反映)
        """
        if not self.trace_mode:
            if self._axonal:
                # nearest × axonal は直接カーネル(st_pre/st_post を直に使用)。post trace 不要。
                return ""
            # nearest: = 1.0 で上書き（蓄積しない = nearest-neighbor）
            if self._is_e_stdp:
                return "post_trace = 1.0;\n"
            return "post_trace1 = 1.0;\npost_trace2 = 1.0;\n"
        if self._is_e_stdp:
            return f"""
                post_trace *= exp(-(t - st_post) / tau_E);
                post_trace += 1.0;
            """
        return f"""
            const scalar dt_post_trace = t - st_post;
            post_trace1 *= exp(-dt_post_trace / tau_I1);
            post_trace2 *= exp(-dt_post_trace / tau_I2);
            post_trace1 += 1.0;
            post_trace2 += 1.0;
        """

    def _build_syn_codes(self):
        """シナプス上の伝播 + 重み更新コードを algorithm でディスパッチする。

        戻り値は (pre_spike_syn_code, post_spike_syn_code, syn_dyn_code,
        pre_arrival_syn_code) の 4 要素タプル。delay_corrected / nearest_dc の非軸索パスは
        毎ステップ syn_dynamics ポーリングをやめ、per-synapse 到着時刻にイベント駆動で走る
        pre_arrival_syn_code を返す (syn_dyn_code=None)。それ以外は pre_arrival_syn_code=None。
        """
        syn_dyn_code = None
        pre_arrival_syn_code = None
        if self.trace_mode:
            if self._axonal:
                # trace × axonal: 到着時刻基準の直接 STDP (per-neuron trace)。
                # GeNN が st_pre を到着時刻へ補正するため遅延補正済みになる。
                pre_spike_syn_code, post_spike_syn_code = self._trace_legacy_syn()
            else:
                # trace × 非軸索(delay_corrected): 到着イベント駆動 (pre_arrival_syn_code)。
                pre_spike_syn_code, post_spike_syn_code, pre_arrival_syn_code = self._trace_dc_syn()
        else:
            if self._axonal:
                # nearest × axonal: 直接指数カーネル版。
                pre_spike_syn_code, post_spike_syn_code = self._nearest_syn()
            else:
                # nearest × 非軸索: 到着イベント駆動 (pre_arrival_syn_code)。
                pre_spike_syn_code, post_spike_syn_code, pre_arrival_syn_code = self._nearest_dc_syn()
        return pre_spike_syn_code, post_spike_syn_code, syn_dyn_code, pre_arrival_syn_code

    def _transmit_stmt(self):
        """シナプス電流の投与文。
        - axonal: 既に到着時刻に pre_spike_syn_code が呼ばれるので addToPost で即時投与。
        - 非axonal: per-synapse 遅延 d ステップ後に届くよう addToPostDelay で据置き投与。
        """
        if self._axonal:
            return "addToPost(x_release * w * g_max * g_scale);\n"
        return "addToPostDelay(x_release * w * g_max * g_scale, d);\n"

    # --- nearest-neighbor (非 trace) ---
    def _nearest_syn(self):
        pre_spike_syn_code = self._transmit_stmt()
        if self._is_e_stdp:
            pre_spike_syn_code += f"""
                const scalar dt = t - st_post;
                if (dt > 0.0) {{
                    const scalar timing = exp(-dt / tau_E);
                    const scalar newWeight = w - (A_E * beta_E * timing);
                    w = fmax(wMin, fmin(wMax, newWeight));
                }}
            """
            post_spike_syn_code = f"""
                const scalar dt = t - st_pre;
                if (dt >= 0.0) {{
                    const scalar timing = exp(-dt / tau_E);
                    const scalar newWeight = w + (A_E * timing);
                    w = fmax(wMin, fmin(wMax, newWeight));
                }}
            """
            return pre_spike_syn_code, post_spike_syn_code

        pre_spike_syn_code += f"""
                const scalar dt = t - st_post;
                if (dt > 0.0) {{
                    const scalar timing1 = exp(-dt / tau_I1);
                    const scalar timing2 = exp(-dt / tau_I2);
                    const scalar dW = C_I * (timing1 - C_I_beta * timing2);
                    w = fmax(wMin, fmin(wMax, w + dW));
                }}
            """
        post_spike_syn_code = f"""
                const scalar dt = t - st_pre;
                if (dt >= 0.0) {{
                    const scalar timing1 = exp(-dt / tau_I1);
                    const scalar timing2 = exp(-dt / tau_I2);
                    const scalar dW = C_I * (timing1 - C_I_beta * timing2);
                    w = fmax(wMin, fmin(wMax, w + dW));
                }}
            """
        return pre_spike_syn_code, post_spike_syn_code

    # --- nearest: 到着イベント駆動 + = 1.0 上書き (nearest-neighbor) ---
    def _nearest_dc_syn(self):
        """nearest モード（非軸索）: per-synapse 到着イベント駆動。trace 更新を += でなく
        = 1.0 で行い、重み計算は常に直近スパイク1本分のみに基づく（nearest-neighbor ペアリング）。

        旧実装は毎ステップ syn_dynamics で到着をポーリングしていたが、GeNN fork の
        pre_arrival_syn_code で到着時刻にのみ実行する。st_pre_axon (per-synapse 到着時刻)
        を使い、LTP は post 発火時に直近到着から現在まで trace を減衰して適用する。
        Δt=0 は GeNN のカーネル順序保証により LTP のみ (到着カーネルは dt_post>0 のみ LTD)。

        戻り値: (pre_spike_syn_code, post_spike_syn_code, pre_arrival_syn_code)
        """
        pre_spike_syn_code = self._transmit_stmt()

        if self._is_e_stdp:
            # LTP: 直近到着 st_pre_axon から現在まで pre_trace を減衰して適用
            post_spike_syn_code = f"""
                        const scalar pre_trace_now = pre_trace_syn * exp(-(t - st_pre_axon) / tau_E);
                        const scalar newWeight = w + (A_E * pre_trace_now);
                        w = fmax(wMin, fmin(wMax, newWeight));
                    """
            # 到着イベント: nearest なので trace を 1.0 上書き + (post が先行していれば) LTD
            pre_arrival_syn_code = f"""
                        pre_trace_syn = 1.0;
                        const scalar dt_post = t - st_post;
                        if (dt_post > 0.0) {{
                            const scalar post_trace_now = post_trace * exp(-dt_post / tau_E);
                            w -= A_E * beta_E * post_trace_now;
                            w = fmax(wMin, fmin(wMax, w));
                        }}
                    """
            return pre_spike_syn_code, post_spike_syn_code, pre_arrival_syn_code

        # I-STDP
        post_spike_syn_code = f"""
                        const scalar pre_trace1_now = pre_trace_syn1 * exp(-(t - st_pre_axon) / tau_I1);
                        const scalar pre_trace2_now = pre_trace_syn2 * exp(-(t - st_pre_axon) / tau_I2);
                        const scalar dW = C_I * (pre_trace1_now - C_I_beta * pre_trace2_now);
                        w = fmax(wMin, fmin(wMax, w + dW));
                    """
        pre_arrival_syn_code = f"""
                        pre_trace_syn1 = 1.0;
                        pre_trace_syn2 = 1.0;
                        const scalar dt_post = t - st_post;
                        if (dt_post > 0.0) {{
                            const scalar post_trace1_now = post_trace1 * exp(-dt_post / tau_I1);
                            const scalar post_trace2_now = post_trace2 * exp(-dt_post / tau_I2);
                            const scalar dW = C_I * (post_trace1_now - C_I_beta * post_trace2_now);
                            w = fmax(wMin, fmin(wMax, w + dW));
                        }}
                    """
        return pre_spike_syn_code, post_spike_syn_code, pre_arrival_syn_code

    # --- trace 型: legacy (遅延非考慮) ---
    def _trace_legacy_syn(self):
        dt_ms = self._dt_ms
        pre_spike_syn_code = self._transmit_stmt()
        if self._is_e_stdp:
            pre_spike_syn_code += f"""
                        const scalar dt_post_trace = t - st_post;
                        if (dt_post_trace > 0.5 * {dt_ms}) {{
                            const scalar post_trace_now = post_trace * exp(-dt_post_trace / tau_E);
                            const scalar newWeight = w - (A_E * beta_E * post_trace_now);
                            w = fmax(wMin, fmin(wMax, newWeight));
                        }}
                    """
            post_spike_syn_code = f"""
                        const scalar dt_pre_trace = t - st_pre;
                        const scalar pre_trace_now = pre_trace * exp(-dt_pre_trace / tau_E);
                        const scalar newWeight = w + (A_E * pre_trace_now);
                        w = fmax(wMin, fmin(wMax, newWeight));
                    """
            return pre_spike_syn_code, post_spike_syn_code

        pre_spike_syn_code += f"""
                        const scalar dt_post_trace = t - st_post;
                        if (dt_post_trace > 0.5 * {dt_ms}) {{
                            const scalar post_trace1_now = post_trace1 * exp(-dt_post_trace / tau_I1);
                            const scalar post_trace2_now = post_trace2 * exp(-dt_post_trace / tau_I2);
                            const scalar dW = C_I * (post_trace1_now - C_I_beta * post_trace2_now);
                            w = fmax(wMin, fmin(wMax, w + dW));
                        }}
                    """
        post_spike_syn_code = f"""
                        const scalar dt_pre_trace = t - st_pre;
                        const scalar pre_trace1_now = pre_trace1 * exp(-dt_pre_trace / tau_I1);
                        const scalar pre_trace2_now = pre_trace2 * exp(-dt_pre_trace / tau_I2);
                        const scalar dW = C_I * (pre_trace1_now - C_I_beta * pre_trace2_now);
                        w = fmax(wMin, fmin(wMax, w + dW));
                    """
        return pre_spike_syn_code, post_spike_syn_code

    # --- trace 型: delay_corrected (per-synapse trace + 到着イベント駆動) ---
    def _trace_dc_syn(self):
        """delay_corrected（非軸索, all-to-all trace）: per-synapse 到着イベント駆動。

        旧実装は毎ステップ syn_dynamics で pre_trace_syn を減衰しつつ到着をポーリングして
        いた。GeNN fork の pre_arrival_syn_code で到着時刻にのみ実行し、trace は decay-on-read
        （前回到着 st_pre_axon から今回到着まで減衰して +1.0）で更新する。LTP は post 発火時に
        直近到着から現在まで減衰して適用（st_pre_axon 使用）。

        st_pre_axon は GeNN 管理の per-synapse 到着時刻で、pre_arrival_syn_code の本文実行後に
        今回の到着時刻 t へ更新される（本文中では「前回の到着時刻」を保持）。学習カーネル順序
        (arrival → postsynaptic) により、Δt=0 は LTP のみが観測する（旧実装の +1.0 手補正は不要）。

        戻り値: (pre_spike_syn_code, post_spike_syn_code, pre_arrival_syn_code)
        """
        # pre_spike_syn_code: 電流送信のみ（遅延 d 後に addToPostDelay で投与）。
        pre_spike_syn_code = self._transmit_stmt()

        if self._is_e_stdp:
            # LTP: pre_trace_syn を直近到着 st_pre_axon から現在まで減衰して適用。
            post_spike_syn_code = f"""
                        const scalar pre_trace_now = pre_trace_syn * exp(-(t - st_pre_axon) / tau_E);
                        const scalar newWeight = w + (A_E * pre_trace_now);
                        w = fmax(wMin, fmin(wMax, newWeight));
                    """
            # 到着イベント: trace を decay-on-read + 1.0 で更新 → post 先行なら LTD。
            pre_arrival_syn_code = f"""
                        pre_trace_syn = pre_trace_syn * exp(-(t - st_pre_axon) / tau_E) + 1.0;
                        const scalar dt_post = t - st_post;
                        if (dt_post > 0.0) {{
                            const scalar post_trace_now = post_trace * exp(-dt_post / tau_E);
                            w -= A_E * beta_E * post_trace_now;
                            w = fmax(wMin, fmin(wMax, w));
                        }}
                    """
            return pre_spike_syn_code, post_spike_syn_code, pre_arrival_syn_code

        # I-STDP
        post_spike_syn_code = f"""
                        const scalar pre_trace1_now = pre_trace_syn1 * exp(-(t - st_pre_axon) / tau_I1);
                        const scalar pre_trace2_now = pre_trace_syn2 * exp(-(t - st_pre_axon) / tau_I2);
                        const scalar dW = C_I * (pre_trace1_now - C_I_beta * pre_trace2_now);
                        w = fmax(wMin, fmin(wMax, w + dW));
                    """
        pre_arrival_syn_code = f"""
                        pre_trace_syn1 = pre_trace_syn1 * exp(-(t - st_pre_axon) / tau_I1) + 1.0;
                        pre_trace_syn2 = pre_trace_syn2 * exp(-(t - st_pre_axon) / tau_I2) + 1.0;
                        const scalar dt_post = t - st_post;
                        if (dt_post > 0.0) {{
                            const scalar post_trace1_now = post_trace1 * exp(-dt_post / tau_I1);
                            const scalar post_trace2_now = post_trace2 * exp(-dt_post / tau_I2);
                            const scalar dW = C_I * (post_trace1_now - C_I_beta * post_trace2_now);
                            w = fmax(wMin, fmin(wMax, w + dW));
                        }}
                    """
        return pre_spike_syn_code, post_spike_syn_code, pre_arrival_syn_code

    # ==========================================
    # 4. プロパティ (BasePlasticityModel の実装)
    # ==========================================
    @property
    def snippet(self): return self._custom_snippet_obj
    @property
    def params(self): return self._params
    @property
    def vars(self): return self._vars
    @property
    def pre_vars(self): return self._pre_vars
    @property
    def post_vars(self): return self._post_vars
