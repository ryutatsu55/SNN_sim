import pygenn
from src.core.registry import NEURON_MODELS
from .base_models import BaseNeuronModel
from .PQN_origin import PQNengine

@NEURON_MODELS.register("PQN_int")
class PQNIntModel(BaseNeuronModel):
    """
    公式の PQNengine (PQN_origin.py) に基づく固定小数点演算版 PQN モデル。
    パーサーエラーを避けるため、ビットシフト(>>)の代わりに定数除算(/)を利用します。
    (コンパイラによって自動的に高速なシフト命令に最適化されます)
    """
    def __init__(self, mode="RSexci", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.engine = PQNengine(mode=mode)
        
        self.bit_f = self.engine.BIT_WIDTH_FRACTIONAL
        self.bit_y = self.engine.BIT_Y_SHIFT # 常に 20
        
        self._params, self._init_vars = self._prepare_genn_data()

    def _prepare_genn_data(self):
        params = {k.upper(): float(v) for k, v in self.engine.Y.items()}
        params["V_THRESH"] = float(4 << self.bit_f)
        
        init_vars = {
            "V": int(self.engine.state_variable_v),
            "N": int(self.engine.state_variable_n),
            "Q": int(self.engine.state_variable_q),
            "U": int(self.engine.state_variable_u),
            "Iext": 0.0 
        }
        return params, init_vars

    @property
    def model_class(self):
        # Python側でシフト幅に対応する除数（スケール）を計算しておく
        f_scale = 1 << self.bit_f
        b_scale = 1 << self.bit_y

        # f-string を使って、定数として C++ コードに直接埋め込む
        sim_code = f"""
        const int64_t v_long = (int64_t)V;
        const int64_t F_SCALE = (int64_t){f_scale};
        const int64_t B_SCALE = (int64_t){b_scale};
        
        // ビットシフト (>>) の代わりに定数除算 (/) を使用
        const int64_t vv = (v_long * v_long) / F_SCALE;
        const int64_t i_fixed = (int64_t)(Iext * (double)F_SCALE);
        """

        # --- dv (膜電位変化量) の計算 ---
        if self.mode == 'PB':
            sim_code += """
            int64_t dv;
            if (V < 0) {
                dv = (V_VV_S * vv / B_SCALE) + (V_V_S * v_long / B_SCALE) + (int64_t)V_C_S + (V_N * (int64_t)N / B_SCALE) + (V_Q * (int64_t)Q / B_SCALE) - (V_U * (int64_t)U / B_SCALE) + (V_I * i_fixed / B_SCALE);
            } else {
                dv = (V_VV_L * vv / B_SCALE) + (V_V_L * v_long / B_SCALE) + (int64_t)V_C_L + (V_N * (int64_t)N / B_SCALE) + (V_Q * (int64_t)Q / B_SCALE) - (V_U * (int64_t)U / B_SCALE) + (V_I * i_fixed / B_SCALE);
            }
            """
        elif self.mode == 'Class2':
            sim_code += """
            int64_t dv;
            if (V < 0) {
                dv = (V_VV_S * vv / B_SCALE) + (V_V_S * v_long / B_SCALE) + (int64_t)V_C_S + (V_N * (int64_t)N / B_SCALE) + (V_I * i_fixed / B_SCALE);
            } else {
                dv = (V_VV_L * vv / B_SCALE) + (V_V_L * v_long / B_SCALE) + (int64_t)V_C_L + (V_N * (int64_t)N / B_SCALE) + (V_I * i_fixed / B_SCALE);
            }
            """
        else: # RS, FS, LTS, IB, EB
            sim_code += """
            int64_t dv;
            if (V < 0) {
                dv = (V_VV_S * vv / B_SCALE) + (V_V_S * v_long / B_SCALE) + (int64_t)V_C_S + (V_N * (int64_t)N / B_SCALE) + (V_Q * (int64_t)Q / B_SCALE) + (V_I * i_fixed / B_SCALE);
            } else {
                dv = (V_VV_L * vv / B_SCALE) + (V_V_L * v_long / B_SCALE) + (int64_t)V_C_L + (V_N * (int64_t)N / B_SCALE) + (V_Q * (int64_t)Q / B_SCALE) + (V_I * i_fixed / B_SCALE);
            }
            """

        # --- dn (回復変数変化量) の計算 ---
        if self.mode in ['LTS', 'IB']:
            sim_code += """
            int64_t dn;
            if (V < (int64_t)RG) {
                dn = (N_VV_S * vv / B_SCALE) + (N_V_S * v_long / B_SCALE) + (int64_t)N_C_S + (N_N * (int64_t)N / B_SCALE);
            } else {
                dn = (N_VV_L * vv / B_SCALE) + (N_V_L * v_long / B_SCALE) + (int64_t)N_C_L + (N_N * (int64_t)N / B_SCALE);
            }
            // LTS/IB 特有の u による eta 制御
            if (U < (int64_t)RU) {
                dn = (dn * (int64_t)N_US) / B_SCALE;
            } else {
                dn = (dn * (int64_t)N_UL) / B_SCALE;
            }
            """
        else:
            sim_code += """
            int64_t dn;
            if (V < (int64_t)RG) {
                dn = (N_VV_S * vv / B_SCALE) + (N_V_S * v_long / B_SCALE) + (int64_t)N_C_S + (N_N * (int64_t)N / B_SCALE);
            } else {
                dn = (N_VV_L * vv / B_SCALE) + (N_V_L * v_long / B_SCALE) + (int64_t)N_C_L + (N_N * (int64_t)N / B_SCALE);
            }
            """

        # --- dq, du (追加変数) の計算 ---
        if self.mode != 'Class2':
            sim_code += """
            int64_t dq;
            if (V < (int64_t)RH) {
                dq = (Q_VV_S * vv / B_SCALE) + (Q_V_S * v_long / B_SCALE) + (int64_t)Q_C_S + (Q_Q * (int64_t)Q / B_SCALE);
            } else {
                dq = (Q_VV_L * vv / B_SCALE) + (Q_V_L * v_long / B_SCALE) + (int64_t)Q_C_L + (Q_Q * (int64_t)Q / B_SCALE);
            }
            """
        
        if self.mode in ['LTS', 'IB', 'PB']:
            sim_code += """
            int64_t du = (U_V * v_long / B_SCALE) + (U_U * (int64_t)U / B_SCALE) + (int64_t)U_C;
            """

        # --- 状態の更新 ---
        sim_code += "V += (int32_t)dv; N += (int32_t)dn;"
        if self.mode != 'Class2':
            sim_code += "Q += (int32_t)dq;"
        if self.mode in ['LTS', 'IB', 'PB']:
            sim_code += "U += (int32_t)du;"

        return pygenn.create_neuron_model(
            f"PQN_Int_{self.mode}",
            params=list(self._params.keys()),
            vars=[
                ("V", "int32_t"), ("N", "int32_t"), 
                ("Q", "int32_t"), ("U", "int32_t"), 
                ("Iext", "scalar")
            ],
            sim_code=sim_code,
            threshold_condition_code="V >= (int32_t)V_THRESH"
        )

    @property
    def params(self): return self._params

    @property
    def initial_vars(self): return self._init_vars