"""AkitaDai 先生モデルベースのカスタム STDP + STP 可塑性モデル。

`CustomAkitaModel` は GeNN の weight-update スニペットを mode に応じて動的生成する:

    mode = e-stdp / i-stdp           ... 興奮性 / 抑制性 STDP
         × (nearest / _trace)        ... 最近接ペア / trace 型 all-to-all
         × (legacy / delay_corrected) ... trace 型のみ。伝播遅延の扱い

加えて STP（短期可塑性: 資源回復 Eq. S11 / 消費 Eq. S12）を全 mode 共通で持つ。

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

    def __init__(self, config, dt, weight, delay, num_pre, num_post):
        super().__init__(config, dt, weight, delay, num_pre, num_post)
        self.mode = self.config.mode
        if not (self.mode.startswith("e-stdp") or self.mode.startswith("i-stdp")):
            raise ValueError(f"Unsupported mode '{self.mode}' for CustomAkitaModel.")
        self.trace_mode = self.mode.endswith("_trace")
        self.trace_algorithm = getattr(self.config, "trace_algorithm", "delay_corrected") if self.trace_mode else "na"
        # 便宜的な mode 判定ヘルパ (興奮性 STDP かどうか)
        self._is_e_stdp = self.mode.startswith("e-stdp")
        # d はタイムステップ単位(uint8)。GeNN の t は ms 単位なので、STDP の時間演算では
        # d を ms に換算する (d * dt_ms)。※ addToPostDelay の遅延引数はタイムステップ単位のため d のまま使う。
        self._dt_ms = float(self.dt)
        self._gmax_scale = calculate_gmax_scale(
            num_synapses=len(self.weight),
            num_post=self.num_post,
            normalize_by_fan_in=bool(getattr(self.config, "normalize_gmax_by_fan_in", False)),
        )

        self._params, self._vars, self._pre_vars, self._post_vars = self._prepare_genn_data()

        cache_key = (self.mode, "trace" if self.trace_mode else "nearest", "g_scale_param", self.trace_algorithm)
        if cache_key not in CustomAkitaModel._snippet_cache:
            # 未登録の場合のみ生成
            CustomAkitaModel._snippet_cache[cache_key] = self._create_snippet()
        self._custom_snippet_obj = CustomAkitaModel._snippet_cache[cache_key]

    # ==========================================
    # 1. パラメータと変数の準備
    # ==========================================
    def _prepare_genn_data(self):
        # 変数設定 (モード共通)
        vars_dict = {
            "w": self.weight.astype('float32'),
            "d": self.delay.astype('uint8')
        }
        pre_vars_dict = {
            "x": np.ones(self.num_pre, dtype='float32'),
            "x_release": np.zeros(self.num_pre, dtype='float32'),
        }
        post_vars_dict = {}

        if self.trace_mode:
            if self._is_e_stdp:
                # delay_corrected: pre_trace を per-synapse vars に置く（到着時刻に更新するため）
                # legacy: per-neuron pre_vars に置く（変更なし）
                if self.trace_algorithm == "delay_corrected":
                    vars_dict["pre_trace_syn"] = np.zeros(len(self.weight), dtype='float64')
                else:
                    pre_vars_dict["pre_trace"] = np.zeros(self.num_pre, dtype='float64')
                post_vars_dict.update({
                    "post_trace": np.zeros(self.num_post, dtype='float64'),
                })
            else:
                if self.trace_algorithm == "delay_corrected":
                    vars_dict["pre_trace_syn1"] = np.zeros(len(self.weight), dtype='float64')
                    vars_dict["pre_trace_syn2"] = np.zeros(len(self.weight), dtype='float64')
                else:
                    pre_vars_dict["pre_trace1"] = np.zeros(self.num_pre, dtype='float64')
                    pre_vars_dict["pre_trace2"] = np.zeros(self.num_pre, dtype='float64')
                post_vars_dict.update({
                    "post_trace1": np.zeros(self.num_post, dtype='float64'),
                    "post_trace2": np.zeros(self.num_post, dtype='float64'),
                })

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
        pre_code, post_code, pre_syn_code, post_syn_code, syn_dyn_code = self._build_cpp_codes()

        safe_mode = self.mode.replace("-", "_")
        alg_suffix = "leg" if self.trace_algorithm == "legacy" else "dc"

        # per-synapse vars: w, d は共通。delay_corrected では pre_trace_syn も per-synapse
        var_defs = [("w", "scalar"), ("d", "uint8_t")]
        pre_var_defs = [("x", "scalar"), ("x_release", "scalar")]
        post_var_defs = []

        if self.trace_mode:
            if self._is_e_stdp:
                if self.trace_algorithm == "delay_corrected":
                    var_defs.append(("pre_trace_syn", "double"))
                else:
                    pre_var_defs.append(("pre_trace", "double", pygenn.VarAccess.READ_WRITE))
                post_var_defs.append(("post_trace", "double", pygenn.VarAccess.READ_WRITE))
            else:
                if self.trace_algorithm == "delay_corrected":
                    var_defs.append(("pre_trace_syn1", "double"))
                    var_defs.append(("pre_trace_syn2", "double"))
                else:
                    pre_var_defs.append(("pre_trace1", "double", pygenn.VarAccess.READ_WRITE))
                    pre_var_defs.append(("pre_trace2", "double", pygenn.VarAccess.READ_WRITE))
                post_var_defs.append(("post_trace1", "double", pygenn.VarAccess.READ_WRITE))
                post_var_defs.append(("post_trace2", "double", pygenn.VarAccess.READ_WRITE))

        snippet = pygenn.create_weight_update_model(
            class_name=f"custom_Akita_{safe_mode}_{alg_suffix}_gscaled",
            params=list(self._params.keys()),
            vars=var_defs,
            pre_vars=pre_var_defs,
            post_vars=post_var_defs,
            pre_spike_code=pre_code,
            post_spike_code=post_code,
            pre_spike_syn_code=pre_syn_code,
            post_spike_syn_code=post_syn_code,
            synapse_dynamics_code=syn_dyn_code
        )

        return snippet

    # ==========================================
    # 3. C++ ロジック生成
    # ==========================================
    def _build_cpp_codes(self):
        """C++のロジックコードを生成する。

        生成物は 5 つの C++ 文字列:
          - pre_spike_code / post_spike_code        : ニューロン発火時の状態更新 (STP 資源・trace)
          - pre_spike_syn_code / post_spike_syn_code: シナプス上での伝播・重み更新
          - syn_dyn_code                            : 毎ステップ実行 (delay_corrected のみ)
        mode (e/i-stdp) × trace/nearest × legacy/delay_corrected で分岐し、
        実際の生成は下記の per-mode ヘルパへ委譲する。
        """
        pre_spike_code = self._pre_spike_code()
        post_spike_code = self._post_spike_code()
        pre_spike_syn_code, post_spike_syn_code, syn_dyn_code = self._build_syn_codes()
        return pre_spike_code, post_spike_code, pre_spike_syn_code, post_spike_syn_code, syn_dyn_code

    def _pre_spike_code(self):
        """プレニューロン発火時: STP 資源更新 + (legacy trace 型のみ) pre trace 更新。
        delay_corrected では pre_trace_syn を syn_dynamics_code で到着時刻に更新するため、
        ここでは trace 更新を行わない。
        """
        pre_spike_code = f"""
            const scalar dt_pre = t - st_pre;
            x = 1.0 - ((1.0 - x) * exp(-dt_pre / tau_rec));
            x_release = U * x;
            x -= x_release;
        """

        if self.trace_mode and self.trace_algorithm == "legacy":
            # legacy のみ per-neuron pre_trace を発火時に更新する
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
        """ポストニューロン発火時: (trace 型なら) post trace 更新。"""
        if not self.trace_mode:
            return ""
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

        戻り値は (pre_spike_syn_code, post_spike_syn_code, syn_dyn_code) の 3 要素タプル。
        delay_corrected 以外では syn_dyn_code = None。
        """
        if self.trace_mode:
            if self.trace_algorithm == "legacy":
                pre_spike_syn_code, post_spike_syn_code = self._trace_legacy_syn()
                syn_dyn_code = None
            else:
                pre_spike_syn_code, post_spike_syn_code, syn_dyn_code = self._trace_dc_syn()
        else:
            pre_spike_syn_code, post_spike_syn_code = self._nearest_syn()
            syn_dyn_code = None
        return pre_spike_syn_code, post_spike_syn_code, syn_dyn_code

    # --- nearest-neighbor (非 trace) ---
    def _nearest_syn(self):
        pre_spike_syn_code = "addToPostDelay(x_release * w * g_max * g_scale, d);\n"
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

    # --- trace 型: legacy (遅延非考慮) ---
    def _trace_legacy_syn(self):
        pre_spike_syn_code = "addToPostDelay(x_release * w * g_max * g_scale, d);\n"
        if self._is_e_stdp:
            pre_spike_syn_code += f"""
                        const scalar dt_post_trace = t - st_post;
                        if (dt_post_trace > 0.0) {{
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
                        if (dt_post_trace > 0.0) {{
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

    # --- trace 型: delay_corrected (per-synapse trace + syn_dynamics で到着時刻更新) ---
    def _trace_dc_syn(self):
        dt_ms = self._dt_ms

        # pre_spike_syn_code: 電流送信のみ。STDP は syn_dynamics_code で到着時刻に実行する。
        pre_spike_syn_code = "addToPostDelay(x_release * w * g_max * g_scale, d);\n"

        if self._is_e_stdp:
            # post_spike_syn_code: pre_trace_syn は到着済みスパイクのみを反映している。
            # GeNN の double-buffering により、post 発火と同時刻の到着 (delta_t=0) は
            # READ バッファに未コミットのため pre_trace_syn=0 になる (シナプス伝播遅延が
            # post 発火を 1 ステップ遅らせた形で検出される)。
            # この場合、到着分 (+1.0) を自前で補完して正しい LTP を適用する。
            # 判定: t_arr = st_pre + d*dt_ms == t (現在のシミュレーション時刻)
            # ※ post_spike_syn_code における st_pre は +dt オフセットなしの raw srcST。
            post_spike_syn_code = f"""
                        const scalar t_arr = st_pre + (d * {dt_ms});
                        scalar ltp_trace;
                        if (fabs(t - t_arr) < 0.5 * {dt_ms}) {{
                            ltp_trace = pre_trace_syn * decay_E + 1.0;
                        }} else {{
                            ltp_trace = pre_trace_syn;
                        }}
                        const scalar newWeight = w + (A_E * ltp_trace);
                        w = fmax(wMin, fmin(wMax, newWeight));
                    """

            # syn_dynamics_code: 毎ステップ実行。
            # Step1: pre_trace_syn を指数減衰（連続減衰で常に最新値を保つ）
            # Step2: 今ステップにスパイクが到着したか検出
            # Step3: 到着したら +1.0 積算 → LTD を到着時刻基準で適用
            syn_dyn_code = f"""
                        pre_trace_syn *= decay_E;
                        const scalar t_arrival = st_pre + (d * {dt_ms});
                        if (fabs(t - t_arrival) < 0.5 * {dt_ms}) {{
                            pre_trace_syn += 1.0;
                            const scalar dt_post = t - st_post;
                            if (dt_post > 0.5 * {dt_ms}) {{
                                const scalar post_trace_now = post_trace * exp(-dt_post / tau_E);
                                w -= A_E * beta_E * post_trace_now;
                                w = fmax(wMin, fmin(wMax, w));
                            }}
                        }}
                    """
            return pre_spike_syn_code, post_spike_syn_code, syn_dyn_code

        # I-STDP
        post_spike_syn_code = f"""
                        const scalar dW = C_I * (pre_trace_syn1 - C_I_beta * pre_trace_syn2);
                        w = fmax(wMin, fmin(wMax, w + dW));
                    """

        syn_dyn_code = f"""
                        pre_trace_syn1 *= decay_I1;
                        pre_trace_syn2 *= decay_I2;
                        const scalar t_arrival = st_pre + (d * {dt_ms});
                        if (fabs(t - t_arrival) < 0.5 * {dt_ms}) {{
                            pre_trace_syn1 += 1.0;
                            pre_trace_syn2 += 1.0;
                            const scalar dt_post = t - st_post;
                            if (dt_post >  0.5 * {dt_ms}) {{
                                const scalar post_trace1_now = post_trace1 * exp(-dt_post / tau_I1);
                                const scalar post_trace2_now = post_trace2 * exp(-dt_post / tau_I2);
                                const scalar dW = C_I * (post_trace1_now - C_I_beta * post_trace2_now);
                                w = fmax(wMin, fmin(wMax, w + dW));
                            }}
                        }}
                    """
        return pre_spike_syn_code, post_spike_syn_code, syn_dyn_code

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
