import pygenn
from src.core.registry import NEURON_MODELS
from .base_models import BaseNeuronModel
from .PQN_origin import PQNengine

@NEURON_MODELS.register("PQN_int")
class PQNIntModel(BaseNeuronModel):
    """
    公式の PQNengine (PQN_origin.py) に基づく固定小数点演算版 PQN モデル。
    各パラメータと係数 Y を利用し、ビットシフト演算による計算を再現します。
    """
    def __init__(self, mode="RSexci", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        # 公式エンジンを初期化して定数と係数を取得
        self.engine = PQNengine(mode=mode)
        
        # 固定小数点のビット幅設定
        self.bit_f = self.engine.BIT_WIDTH_FRACTIONAL
        self.bit_y = self.engine.BIT_Y_SHIFT # 常に 20
        
        self._params, self._init_vars = self._prepare_genn_data()

    def _prepare_genn_data(self):
        # 公式コードの係数辞書 Y を GeNN のパラメータ用に変換 (大文字化)
        params = {k.upper(): float(v) for k, v in self.engine.Y.items()}
        
        # 制御用のメタパラメータを追加
        params["BIT_F"] = float(self.bit_f)
        params["BIT_Y"] = float(self.bit_y)
        # 閾値は通常 4.0 固定 (固定小数点へ変換)
        params["V_THRESH"] = float(4 << self.bit_f)
        
        # 公式コードに定義されている初期状態変数を取得
        init_vars = {
            "V": int(self.engine.state_variable_v),
            "N": int(self.engine.state_variable_n),
            "Q": int(self.engine.state_variable_q),
            "U": int(self.engine.state_variable_u),
            "Iext": 0.0 # 外部入力は GeNN 側で scalar (float) として受ける
        }
        
        return params, init_vars

    @property
    def model_class(self):
        # 共通の事前計算 (V^2 の算出と Iext の固定小数点化)
        # 符号付き 64bit 整数 (int64_t) を使用してオーバーフローを防止
        sim_code = """
        const int64_t v_long = (int64_t)V;
        const int64_t vv = (v_long * v_long) >> (int)BIT_F;
        const int B = (int)BIT_Y;
        const int64_t i_fixed = (int64_t)(Iext * (double)(1 << (int)BIT_F));
        """

        # --- dv (膜電位変化量) の計算 ---
        if self.mode == 'PB':
            sim_code += """
            int64_t dv;
            if (V < 0) {
                dv = (V_VV_S * vv >> B) + (V_V_S * v_long >> B) + (int64_t)V_C_S + (V_N * (int64_t)N >> B) + (V_Q * (int64_t)Q >> B) - (V_U * (int64_t)U >> B) + (V_I * i_fixed >> B);
            } else {
                dv = (V_VV_L * vv >> B) + (V_V_L * v_long >> B) + (int64_t)V_C_L + (V_N * (int64_t)N >> B) + (V_Q * (int64_t)Q >> B) - (V_U * (int64_t)U >> B) + (V_I * i_fixed >> B);
            }
            """
        elif self.mode == 'Class2':
            sim_code += """
            int64_t dv;
            if (V < 0) {
                dv = (V_VV_S * vv >> B) + (V_V_S * v_long >> B) + (int64_t)V_C_S + (V_N * (int64_t)N >> B) + (V_I * i_fixed >> B);
            } else {
                dv = (V_VV_L * vv >> B) + (V_V_L * v_long >> B) + (int64_t)V_C_L + (V_N * (int64_t)N >> B) + (V_I * i_fixed >> B);
            }
            """
        else: # RS, FS, LTS, IB, EB [cite: 198, 203]
            sim_code += """
            int64_t dv;
            if (V < 0) {
                dv = (V_VV_S * vv >> B) + (V_V_S * v_long >> B) + (int64_t)V_C_S + (V_N * (int64_t)N >> B) + (V_Q * (int64_t)Q >> B) + (V_I * i_fixed >> B);
            } else {
                dv = (V_VV_L * vv >> B) + (V_V_L * v_long >> B) + (int64_t)V_C_L + (V_N * (int64_t)N >> B) + (V_Q * (int64_t)Q >> B) + (V_I * i_fixed >> B);
            }
            """

        # --- dn (回復変数変化量) の計算 ---
        if self.mode in ['LTS', 'IB']:
            sim_code += """
            int64_t dn;
            if (V < (int64_t)RG) {
                dn = (N_VV_S * vv >> B) + (N_V_S * v_long >> B) + (int64_t)N_C_S + (N_N * (int64_t)N >> B);
            } else {
                dn = (N_VV_L * vv >> B) + (N_V_L * v_long >> B) + (int64_t)N_C_L + (N_N * (int64_t)N >> B);
            }
            // LTS/IB 特有の u による eta 制御 [cite: 236-239]
            if (U < (int64_t)RU) {
                dn = (dn * (int64_t)N_US) >> B;
            } else {
                dn = (dn * (int64_t)N_UL) >> B;
            }
            """
        else:
            sim_code += """
            int64_t dn;
            if (V < (int64_t)RG) {
                dn = (N_VV_S * vv >> B) + (N_V_S * v_long >> B) + (int64_t)N_C_S + (N_N * (int64_t)N >> B);
            } else {
                dn = (N_VV_L * vv >> B) + (N_V_L * v_long >> B) + (int64_t)N_C_L + (N_N * (int64_t)N >> B);
            }
            """

        # --- dq, du (追加変数) の計算 ---
        if self.mode != 'Class2':
            sim_code += """
            int64_t dq;
            if (V < (int64_t)RH) {
                dq = (Q_VV_S * vv >> B) + (Q_V_S * v_long >> B) + (int64_t)Q_C_S + (Q_Q * (int64_t)Q >> B);
            } else {
                dq = (Q_VV_L * vv >> B) + (Q_V_L * v_long >> B) + (int64_t)Q_C_L + (Q_Q * (int64_t)Q >> B);
            }
            """
        
        if self.mode in ['LTS', 'IB', 'PB']:
            sim_code += """
            int64_t du = (U_V * v_long >> B) + (U_U * (int64_t)U >> B) + (int64_t)U_C;
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