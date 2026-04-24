import pygenn
import numpy as np
from src.core.registry import PLASTICITY_MODELS
from .BASE_plasticity import BasePlasticityModel

@PLASTICITY_MODELS.register("custom_Akita")
class CustomAkitaModel(BasePlasticityModel):
    """
    AkitaDai先生のモデルをベースにしたカスタムSTDP+STPモデル。
    プレ発火時のLTDとポスト発火時のLTPを、Akita論文の式に基づいて実装。
    STPは、プレ発火ごとにリソースxが消費され、時間経過で回復する単純なモデル。
    """
    _snippet_cache = {}
    def __init__(self, config, dt, weight, delay):
        super().__init__(config, dt, weight, delay)
        self._params, self._vars, self._pre_vars, self._post_vars = self._prepare_genn_data()
        self._model_class_instance = self._get_or_create_snippet()

    def _prepare_genn_data(self):
        if self.config.mode == "e-stdp":
            params = {
                "tau_rec": float(self.config.tau_rec),
                "U": float(self.config.U),
                "g_max": float(self.config.g_max),
                "A_E": float(self.config.A_E),
                "tau_E": float(self.config.tau_E),
                "beta_E": float(self.config.beta_E),
                "wMin": float(self.config.Wmin),
                "wMax": float(self.config.Wmax)
            }
        elif self.config.mode == "i-stdp":
            tau_I1 = float(self.config.tau_I1)
            tau_I2 = float(self.config.tau_I2)
            beta_I = float(self.config.beta_I)
            A_I = float(self.config.A_I)
            
            C_I_beta = (tau_I1 / tau_I2) * beta_I
            C_I = A_I / (1.0 - C_I_beta)

            params = {
                "tau_rec": float(self.config.tau_rec),
                "U": float(self.config.U),
                "g_max": float(self.config.g_max),
                "tau_I1": tau_I1,
                "tau_I2": tau_I2,
                "C_I": C_I,
                "C_I_beta": C_I_beta,
                "wMin": float(self.config.Wmin),
                "wMax": float(self.config.Wmax)
            }
        else:
            raise ValueError(f"Unsupported mode '{self.config.mode}' for CustomAkitaModel.")
        vars = {
            "g": self.weight.astype('float32'),
            "d": self.delay.astype('uint8')
        }
        pre_vars = {
            "x": np.ones_like(self.weight, dtype='float32'),  # 初期リソースは全て1
            "t_last_pre": np.full_like(self.weight, -1e9, dtype='float32')  # 最終発火時刻の初期値
        }
        post_vars = {}  # 今回はポストニューロン変数なし

        return params, vars, pre_vars, post_vars
    
    def _get_or_create_snippet(self):
        """スニペットを生成するか、キャッシュから返す"""
        mode = self.config.mode
        
        # すでに同じモードのモデルが生成されていれば、それを返す（二重登録エラーも防げる）
        if mode in CustomAkitaModel._snippet_cache:
            return CustomAkitaModel._snippet_cache[mode]

        # ① プレニューロン発火時に1度だけ実行 (リソースxの回復と消費)
        pre_spike_code = f"""
            x = x * (1.0 - {self.config.U});
            const scalar dt_pre = t - t_last_pre;
            x = 1.0 - ((1.0 - x) * exp(-dt_pre / {self.config.tau_rec}));
            t_last_pre = t;
        """
        # ② プレ発火時に「各シナプス」で実行 (伝播とLTD)
        pre_spike_syn_code="""
            // 1. 伝播: 消費した分のリソースをポストへ注入 (Eq. S9)
            addToPostDelay(U * x * g * g_max, d);
        """
        if self.config.mode == "e-stdp":
            pre_spike_syn_code += """
                // 2. LTD (Depression): Pre発火時に、過去のPost発火を参照する
                const scalar dt = t - st_post; 
                if (dt > 0.0) {
                    const scalar timing = exp(-dt / tau_E);
                    // Akitaモデルの t < 0 (Pre after Post) の式: -A_E * beta_E * exp(t/tau_E)
                    const scalar newWeight = g - (A_E * beta_E * timing);
                    g = fmax(wMin, fmin(wMax, newWeight));
                }
            """
        elif self.config.mode == "i-stdp":
            pre_spike_syn_code += """
                // 2. Pre発火時に、過去のPost発火を参照する
                const scalar dt = t - st_post; 
                if (dt > 0.0) {
                    const scalar timing1 = exp(-dt / tau_I1);
                    const scalar timing2 = exp(-dt / tau_I2);
                    const scalar dW = C_I * (timing1 - C_I_beta * timing2);
                    
                    const scalar newWeight = g + dW;
                    g = fmax(wMin, fmin(wMax, newWeight));
                }
            """
        else:
            raise ValueError(f"Unsupported mode '{self.config.mode}' for CustomAkitaModel.")
        
        if self.config.mode == "e-stdp":
            # ③ ポスト発火時に「各シナプス」で実行 (LTP)
            post_spike_syn_code="""
                // 3. LTP (Potentiation): Post発火時に、過去のPre発火を参照する
                const scalar dt = t - st_pre;
                if (dt >= 0.0) {
                    const scalar timing = exp(-dt / tau_E);
                    // Akitaモデルの t >= 0 (Post after Pre) の式: A_E * exp(-t/tau_E)
                    const scalar newWeight = g + (A_E * timing);
                    g = fmax(wMin, fmin(wMax, newWeight));
                }
            """
        elif self.config.mode == "i-stdp":
            post_spike_syn_code="""
                // 3. Post発火時に、過去のPre発火を参照する
                const scalar dt = t - st_pre;
                if (dt > 0.0) {
                    // 関数が対称(|t|に依存)なため、Pre側と全く同じ計算を行う
                    const scalar timing1 = exp(-dt / tau_I1);
                    const scalar timing2 = exp(-dt / tau_I2);
                    const scalar dW = C_I * (timing1 - C_I_beta * timing2);
                    
                    const scalar newWeight = g + dW;
                    g = fmax(wMin, fmin(wMax, newWeight));
                }
            """
        else:
            raise ValueError(f"Unsupported mode '{self.config.mode}' for CustomAkitaModel.")

        snippet=pygenn.create_weight_update_model(
            class_name=f"custom_Akita_{self.config.mode}",
            params=list(self._params.keys()),
            vars=[("g", "scalar"), ("d", "uint8_t")],
            pre_vars=[("x", "scalar"), ("t_last_pre", "scalar")],
            pre_spike_code=pre_spike_code,
            pre_spike_syn_code=pre_spike_syn_code,
            post_spike_syn_code=post_spike_syn_code
        )
        
        # キャッシュに保存して寿命を永続化
        CustomAkitaModel._snippet_cache[mode] = snippet
        return snippet
    @property
    def model_class(self):
        return self._get_or_create_snippet()
    
    @property
    def params(self): return self._params

    @property
    def vars(self): return self._vars

    @property
    def pre_vars(self): return self._pre_vars
    
    @property
    def post_vars(self): return self._post_vars