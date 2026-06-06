import pygenn
import numpy as np
from src.core.registry import PLASTICITY_MODELS
from triton.language import const
from .BASE_plasticity import BasePlasticityModel

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
        self._gmax_scale = calculate_gmax_scale(
            num_synapses=len(self.weight),
            num_post=self.num_post,
            normalize_by_fan_in=bool(getattr(self.config, "normalize_gmax_by_fan_in", False)),
        )

        self._params, self._vars, self._pre_vars, self._post_vars = self._prepare_genn_data()

        cache_key = (self.mode, "trace" if self.trace_mode else "nearest", "g_scale_param")
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
            "t_last_pre": np.full(self.num_pre, -1e9, dtype='float64')
        }
        post_vars_dict = {}

        if self.trace_mode:
            if self.mode.startswith("e-stdp"):
                pre_vars_dict.update({
                    "pre_trace": np.zeros(self.num_pre, dtype='float64'),
                })
                post_vars_dict.update({
                    "post_trace": np.zeros(self.num_post, dtype='float64'),
                })
            else:
                pre_vars_dict.update({
                    "pre_trace1": np.zeros(self.num_pre, dtype='float64'),
                    "pre_trace2": np.zeros(self.num_pre, dtype='float64'),
                })
                post_vars_dict.update({
                    "post_trace1": np.zeros(self.num_post, dtype='float64'),
                    "post_trace2": np.zeros(self.num_post, dtype='float64'),
                })
        
        # モードに応じたパラメータ取得メソッドへディスパッチ
        if self.mode.startswith("e-stdp"):
            params = self._get_e_stdp_params()
        else:
            params = self._get_i_stdp_params()

        return params, vars_dict, pre_vars_dict, post_vars_dict

    def _get_e_stdp_params(self):
        return {
            "tau_rec": float(self.config.tau_rec),
            "U": float(self.config.U),
            "g_max": float(self.config.g_max),
            "g_scale": float(self._gmax_scale),
            "A_E": float(self.config.A_E),
            "tau_E": float(self.config.tau_E),
            "beta_E": float(self.config.beta_E),
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
            "wMin": float(self.config.Wmin),
            "wMax": float(self.config.Wmax)
        }

    def _create_snippet(self):
        pre_code, post_code, pre_syn_code, post_syn_code = self._build_cpp_codes()

        safe_mode = self.mode.replace("-", "_")
        pre_var_defs = [("x", "scalar"), ("x_release", "scalar"), ("t_last_pre", "double")]
        post_var_defs = []
        if self.trace_mode:
            if self.mode.startswith("e-stdp"):
                pre_var_defs.extend([
                    ("pre_trace", "double", pygenn.VarAccess.READ_WRITE),
                ])
                post_var_defs.extend([
                    ("post_trace", "double", pygenn.VarAccess.READ_WRITE),
                ])
            else:
                pre_var_defs.extend([
                    ("pre_trace1", "double", pygenn.VarAccess.READ_WRITE),
                    ("pre_trace2", "double", pygenn.VarAccess.READ_WRITE),
                ])
                post_var_defs.extend([
                    ("post_trace1", "double", pygenn.VarAccess.READ_WRITE),
                    ("post_trace2", "double", pygenn.VarAccess.READ_WRITE),
                ])

        snippet = pygenn.create_weight_update_model(
            class_name=f"custom_Akita_{safe_mode}_gscaled",
            params=list(self._params.keys()),
            vars=[("w", "scalar"), ("d", "uint8_t")],
            pre_vars=pre_var_defs,
            post_vars=post_var_defs,
            pre_spike_code=pre_code,
            post_spike_code=post_code,
            pre_spike_syn_code=pre_syn_code,
            post_spike_syn_code=post_syn_code
        )

        return snippet

    def _build_cpp_codes(self):
        """C++のロジックコードを生成する"""
        cfg = self.config
        # d はタイムステップ単位(uint8)。GeNN の t は ms 単位なので、
        # STDP の時間演算では d を ms に換算する必要がある (d * dt)。
        # ※ addToPostDelay の遅延引数はタイムステップ単位のため d のまま使う。
        dt_ms = float(self.dt)

        # プレニューロン発火時: Eq. S11で回復し、Eq. S12で放出分を消費する。
        pre_spike_code = f"""
            const scalar dt_pre = t - t_last_pre;
            x = 1.0 - ((1.0 - x) * exp(-dt_pre / {cfg.tau_rec}));
            x_release = {cfg.U} * x;
            x -= x_release;
            t_last_pre = t;
        """

        post_spike_code = ""

        if self.trace_mode:
            if self.mode.startswith("e-stdp"):
                pre_spike_code += f"""
                    pre_trace *= exp(-(t - st_pre) / {cfg.tau_E});
                    pre_trace += 1.0;
                """
                post_spike_code = f"""
                    post_trace *= exp(-(t - st_post) / {cfg.tau_E});
                    post_trace += 1.0;
                """
            else:
                pre_spike_code += f"""
                    const scalar dt_pre_trace = t - st_pre;
                    pre_trace1 *= exp(-dt_pre_trace / {cfg.tau_I1});
                    pre_trace2 *= exp(-dt_pre_trace / {cfg.tau_I2});
                    pre_trace1 += 1.0;
                    pre_trace2 += 1.0;
                """
                post_spike_code = f"""
                    const scalar dt_post_trace = t - st_post;
                    post_trace1 *= exp(-dt_post_trace / {cfg.tau_I1});
                    post_trace2 *= exp(-dt_post_trace / {cfg.tau_I2});
                    post_trace1 += 1.0;
                    post_trace2 += 1.0;
                """
        
        # 伝播の基本コード (共通)
        pre_spike_syn_code = "addToPostDelay(x_release * w * g_max * g_scale, d);\n"

        if self.trace_mode:
            if self.mode.startswith("e-stdp"):
                # Pre発火時: 過去のPost発火に対するLTD計算 (未来のPost発火は知り得ないのでここでは計算しない)
                pre_spike_syn_code += f"""
                    const scalar dt_post = t + (d * {dt_ms}) - st_post;
                    const scalar post_trace_now = post_trace * exp(-dt_post / tau_E);
                    const scalar newWeight = w - (A_E * beta_E * post_trace_now);
                    w = fmax(wMin, fmin(wMax, newWeight));
                """
                # Post発火時: PreからのLTP計算 ＋ 伝播中のPreスパイクが取りこぼしたLTDの補填
                post_spike_syn_code = f"""
                    const scalar dt_pre_arrival = t - (d * {dt_ms}) - st_pre;
                    scalar newWeight = w;

                    if (dt_pre_arrival < 0.0) {{
                        // 【ロールバック】伝播中のPreスパイクが存在する場合
                        // 1. 伝播中のスパイク(+1.0)を除外して過去のLTPを計算
                        const scalar pre_trace_restored = (pre_trace - 1.0) * exp(-dt_pre_arrival / tau_E);
                        newWeight += A_E * pre_trace_restored;
    
                        // 2. 伝播中のPreスパイクが到着時に行うはずだったLTD計算を、今のpost_traceを使って先取り補填
                        //    ※ t == t_last_pre_trace（pre/postの同時発火）は pre_spike_syn 側で
                        //      既にLTD計上済みのため、ここで補填すると二重計上になる → 厳密にpost後発のみ補填
                        if (t > st_pre) {{
                            const scalar time_until_arrival = st_pre + (d * {dt_ms}) - t;
                            const scalar missing_LTD = A_E * beta_E * exp(-time_until_arrival / tau_E);
                            newWeight -= missing_LTD;
                        }}
                    }} else {{
                        // 通常のLTP計算
                        const scalar pre_trace_now = pre_trace * exp(-dt_pre_arrival / tau_E);
                        newWeight += A_E * pre_trace_now;
                    }}
                    w = fmax(wMin, fmin(wMax, newWeight));
                """
            else:
                pre_spike_syn_code += f"""
                    const scalar dt_post_trace = t + (d * {dt_ms}) - st_post;
                    const scalar post_trace1_now = post_trace1 * exp(-dt_post_trace / tau_I1);
                    const scalar post_trace2_now = post_trace2 * exp(-dt_post_trace / tau_I2);
                    const scalar dW = C_I * (post_trace1_now - C_I_beta * post_trace2_now);
                    w = fmax(wMin, fmin(wMax, w + dW));
                """
                post_spike_syn_code = f"""
                    const scalar dt_pre_arrival = t - (d * {dt_ms}) - st_pre;
                    scalar newWeight = w;

                    if (dt_pre_arrival < 0.0) {{
                        // 【ロールバック】
                        // 1. 伝播中のスパイク(+1.0)を除外した更新
                        const scalar pre_trace1_restored = (pre_trace1 - 1.0) * exp(-dt_pre_arrival / tau_I1);
                        const scalar pre_trace2_restored = (pre_trace2 - 1.0) * exp(-dt_pre_arrival / tau_I2);
                        newWeight += C_I * (pre_trace1_restored - C_I_beta * pre_trace2_restored);

                        // 2. 伝播中のPreスパイクの先取り補填
                        //    ※ t == t_last_pre_trace（pre/postの同時発火）は pre_spike_syn 側で
                        //      既に計上済みのため、ここで補填すると二重計上になる → 厳密にpost後発のみ補填
                        if (t > st_pre) {{
                            const scalar time_until_arrival = st_pre + (d * {dt_ms}) - t;
                            const scalar future_post1 = post_trace1 * exp(-time_until_arrival / tau_I1);
                            const scalar future_post2 = post_trace2 * exp(-time_until_arrival / tau_I2);
                            newWeight += C_I * (future_post1 - C_I_beta * future_post2);
                        }}
                    }} else {{
                        // 通常の更新
                        const scalar pre_trace1_now = pre_trace1 * exp(-dt_pre_arrival / tau_I1);
                        const scalar pre_trace2_now = pre_trace2 * exp(-dt_pre_arrival / tau_I2);
                        newWeight += C_I * (pre_trace1_now - C_I_beta * pre_trace2_now);
                    }}
                    w = fmax(wMin, fmin(wMax, newWeight));
                """
        elif self.mode.startswith("e-stdp"):
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
        else: # i-stdp
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
            
        return pre_spike_code, post_spike_code, pre_spike_syn_code, post_spike_syn_code

    # ==========================================
    # 3. プロパティ (BasePlasticityModel の実装)
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
    
