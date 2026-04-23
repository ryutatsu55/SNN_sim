import pygenn
from src.core.registry import NEURON_MODELS
from .base_models import BaseNeuronModel
from .PQN_origin import PQNengine

@NEURON_MODELS.register("PQN_int")
class PQNIntModel(BaseNeuronModel):
    """
    公式の PQNengine に基づく固定小数点演算版 PQN モデル。
    GeNNパーサーエラーを回避しつつ、float変換による誤差と負数のシフト丸め誤差を完全に排除した設計。
    """
    def __init__(self, config, dt):
        super().__init__(config, dt)
        self.engine = PQNengine(mode=config.mode)
        self.bit_f = self.engine.BIT_WIDTH_FRACTIONAL
        self.bit_y = self.engine.BIT_Y_SHIFT 
        
        self._params, self._init_vars = self._prepare_genn_data()

    def _prepare_genn_data(self):
        # ⚠️パラメータは全て C++ 側に定数として直接埋め込むため、GeNN の params は空にする。
        # これにより、float 型への暗黙キャストによる下位ビットの量子化誤差を完全に排除。
        params = {}
        
        init_vars = {
            "V": int(self.engine.state_variable_v),
            "V_prev": int(self.engine.state_variable_v),
            "N": int(self.engine.state_variable_n),
            "Q": int(self.engine.state_variable_q),
            "U": int(self.engine.state_variable_u),
            "Iext": 0.0
        }
        return params, init_vars

    @property
    def model_class(self):
        # Pythonの算術右シフト(床丸め)を、C++の除算と三項演算子で完全再現するインライン展開関数
        # #defineや>>を使わないため、GeNNのパーサーを安全に通過する
        def SR(expr):
            return f"(({expr}) < 0 ? (({expr}) - B_MASK) / B_SCALE : ({expr}) / B_SCALE)"

        # 公式エンジンの Y 係数を全て int64_t の C++ 定数宣言として文字列生成
        y_consts = "\n".join([f"const int64_t {k.upper()} = (int64_t){v};" for k, v in self.engine.Y.items()])

        sim_code = f"""
        V_prev = V;
        // === パラメータの完全定数化 (float変換排除) ===
        {y_consts}

        const scalar I_total = Iext + Isyn;
        
        const int64_t v_long = (int64_t)V;
        const int64_t F_SCALE = (int64_t){1 << self.bit_f};
        const int64_t B_SCALE = (int64_t){1 << self.bit_y};
        const int64_t B_MASK  = (int64_t){(1 << self.bit_y) - 1};

        // 二乗は必ず正になるため通常の除算で丸め誤差なし
        const int64_t vv = (v_long * v_long) / F_SCALE;
        // Iext スケーリング時のみ C++ の double キャスト(ゼロ方向切り捨て)で Python の int() と一致
        const int64_t i_fixed = (int64_t)(I_total * (double)F_SCALE);
        """

        # --- dv (膜電位変化量) の計算 ---
        if self.config.mode == 'PB':
            sim_code += f"""
            int64_t dv;
            if (V < 0) {{
                dv = {SR('V_VV_S * vv')} + {SR('V_V_S * v_long')} + V_C_S + {SR('V_N * (int64_t)N')} + {SR('V_Q * (int64_t)Q')} - {SR('V_U * (int64_t)U')} + {SR('V_I * i_fixed')};
            }} else {{
                dv = {SR('V_VV_L * vv')} + {SR('V_V_L * v_long')} + V_C_L + {SR('V_N * (int64_t)N')} + {SR('V_Q * (int64_t)Q')} - {SR('V_U * (int64_t)U')} + {SR('V_I * i_fixed')};
            }}
            """
        elif self.config.mode == 'Class2':
            sim_code += f"""
            int64_t dv;
            if (V < 0) {{
                dv = {SR('V_VV_S * vv')} + {SR('V_V_S * v_long')} + V_C_S + {SR('V_N * (int64_t)N')} + {SR('V_I * i_fixed')};
            }} else {{
                dv = {SR('V_VV_L * vv')} + {SR('V_V_L * v_long')} + V_C_L + {SR('V_N * (int64_t)N')} + {SR('V_I * i_fixed')};
            }}
            """
        else: # RS, FS, LTS, IB, EB
            sim_code += f"""
            int64_t dv;
            if (V < 0) {{
                dv = {SR('V_VV_S * vv')} + {SR('V_V_S * v_long')} + V_C_S + {SR('V_N * (int64_t)N')} + {SR('V_Q * (int64_t)Q')} + {SR('V_I * i_fixed')};
            }} else {{
                dv = {SR('V_VV_L * vv')} + {SR('V_V_L * v_long')} + V_C_L + {SR('V_N * (int64_t)N')} + {SR('V_Q * (int64_t)Q')} + {SR('V_I * i_fixed')};
            }}
            """

        # --- dn (回復変数変化量) の計算 ---
        if self.config.mode in ['LTS', 'IB']:
            sim_code += f"""
            int64_t dn;
            if (V < RG) {{
                dn = {SR('N_VV_S * vv')} + {SR('N_V_S * v_long')} + N_C_S + {SR('N_N * (int64_t)N')};
            }} else {{
                dn = {SR('N_VV_L * vv')} + {SR('N_V_L * v_long')} + N_C_L + {SR('N_N * (int64_t)N')};
            }}
            if (U < RU) {{
                dn = {SR('dn * N_US')};
            }} else {{
                dn = {SR('dn * N_UL')};
            }}
            """
        else:
            sim_code += f"""
            int64_t dn;
            if (V < RG) {{
                dn = {SR('N_VV_S * vv')} + {SR('N_V_S * v_long')} + N_C_S + {SR('N_N * (int64_t)N')};
            }} else {{
                dn = {SR('N_VV_L * vv')} + {SR('N_V_L * v_long')} + N_C_L + {SR('N_N * (int64_t)N')};
            }}
            """

        # --- dq, du (追加変数) の計算 ---
        if self.config.mode != 'Class2':
            sim_code += f"""
            int64_t dq;
            if (V < RH) {{
                dq = {SR('Q_VV_S * vv')} + {SR('Q_V_S * v_long')} + Q_C_S + {SR('Q_Q * (int64_t)Q')};
            }} else {{
                dq = {SR('Q_VV_L * vv')} + {SR('Q_V_L * v_long')} + Q_C_L + {SR('Q_Q * (int64_t)Q')};
            }}
            """
        
        if self.config.mode in ['LTS', 'IB', 'PB']:
            sim_code += f"""
            int64_t du = {SR('U_V * v_long')} + {SR('U_U * (int64_t)U')} + U_C;
            """

        # --- 状態の更新 ---
        sim_code += "V += (int32_t)dv; N += (int32_t)dn;"
        if self.config.mode != 'Class2':
            sim_code += "Q += (int32_t)dq;"
        if self.config.mode in ['LTS', 'IB', 'PB']:
            sim_code += "U += (int32_t)du;"

        return pygenn.create_neuron_model(
            f"PQN_Int_{self.config.mode}",
            params=list(self._params.keys()),
            vars=[
                ("V", "int32_t"), 
                ("V_prev", "int32_t"),
                ("N", "int32_t"), 
                ("Q", "int32_t"), 
                ("U", "int32_t"), 
                ("Iext", "scalar")
            ],
            sim_code=sim_code,
            threshold_condition_code=f"V >= (int32_t){4 << self.bit_f} && V_prev < (int32_t){4 << self.bit_f}"
        )

    @property
    def params(self): return self._params

    @property
    def initial_vars(self): return self._init_vars