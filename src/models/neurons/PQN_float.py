import pygenn
from src.core.registry import NEURON_MODELS
from .base_models import BaseNeuronModel
from .PQN_origin import PQNengine

@NEURON_MODELS.register("PQN_float")
class PQNFloatModel(BaseNeuronModel):
    """
    浮動小数点演算（FP32）による PQN モデル。
    全モードに対応し、オイラー法を用いてダイナミクスを計算します。
    """
    def __init__(self, config, dt):
        super().__init__(config, dt)
        self.dt = dt
        self.engine = PQNengine(mode=config.mode)
        self._params, self._init_vars = self._prepare_genn_data()

    def _prepare_genn_data(self):
        p = self.engine.PARAM
        
        # 実行時の計算回数を減らすため、あらかじめ Python 側で定数化できる項を計算
        # 例: PHI / TAU や 1.0 / TAU など
        params = {
            "PHI_OVER_TAU": float(p['phi'] / p['tau']),
            "INV_TAU": float(1.0 / p['tau']),
            "EPSQ_OVER_TAU": float(p['epsq'] / p['tau']),
            "I0": float(p['I0']),
            "K": float(p['k']),
            "A_FN": float(p['afn']), "B_FN": float(p['bfn']), "C_FN": float(p['cfn']),
            "A_FP": float(p['afp']), "B_FP": float(p['bfp']), "C_FP": float(p['cfp']),
            "A_GN": float(p['agn']), "B_GN": float(p['bgn']), "C_GN": float(p['cgn']),
            "A_GP": float(p['agp']), "B_GP": float(p['bgp']), "C_GP": float(p['cgp']),
            "RG": float(p['rg']),
            "A_HN": float(p['ahn']), "B_HN": float(p['bhn']), "C_HN": float(p['chn']),
            "A_HP": float(p['ahp']), "B_HP": float(p['bhp']), "C_HP": float(p['chp']),
            "RH": float(p['rh']),
            "V_THRESH": float(p.get('v_thresh', 4.0))
        }

        # LTS, IB, PB モード用の追加パラメータ
        if self.config.mode in ['LTS', 'IB', 'PB']:
            params.update({
                "EPSU_OVER_TAU": float(p.get('epsu', 0.0) / p['tau']),
                "ALPU": float(p.get('alpu', 0.0)),
                "V0": float(p.get('v0', 0.0)),
                "ETA0": float(p.get('eta0', 1.0)),
                "ETA1": float(p.get('eta1', 1.0)),
                "RU": float(p.get('ru', 0.0))
            })

        # 初期状態変数の設定 (float)
        # PQN_origin の整数初期値を、小数ビット幅 (10 or 20) で割って float に戻す
        scale = 1 << self.engine.BIT_WIDTH_FRACTIONAL
        init_vars = {
            "V": float(self.engine.state_variable_v / scale),
            "N": float(self.engine.state_variable_n / scale),
            "Q": float(self.engine.state_variable_q / scale),
            "U": float(self.engine.state_variable_u / scale),
            "Iext": 0.0
        }
        return params, init_vars

    @property
    def model_class(self):
        # 共通の Piecewise Quadratic 関数 (f, g, h) の定義
        # fma (Fused Multiply-Add) を使用して計算精度と速度を向上
        sim_code = """
        const scalar v_curr = V;
        const scalar f_v = (v_curr < 0.0) ? fma(A_FN, (v_curr - B_FN) * (v_curr - B_FN), C_FN) 
                                         : fma(A_FP, (v_curr - B_FP) * (v_curr - B_FP), C_FP);
        const scalar g_v = (v_curr < RG) ? fma(A_GN, (v_curr - B_GN) * (v_curr - B_GN), C_GN)
                                         : fma(A_GP, (v_curr - B_GP) * (v_curr - B_GP), C_GP);
        """

        # モードに応じた微分方程式の構築
        # --- dV/dt ---
        if self.config.mode == 'PB':
            sim_code += "scalar dV = PHI_OVER_TAU * (f_v - N - Q - U + I0 + K * Iext);"
        elif self.config.mode == 'Class2':
            sim_code += "scalar dV = PHI_OVER_TAU * (f_v - N + I0 + K * Iext);"
        else:
            sim_code += "scalar dV = PHI_OVER_TAU * (f_v - N - Q + I0 + K * Iext);"

        # --- dn/dt ---
        if self.config.mode in ['LTS', 'IB']:
            sim_code += "scalar eta = (U < RU) ? ETA0 : ETA1;"
            sim_code += "scalar dN = INV_TAU * (g_v - N) * eta;"
        else:
            sim_code += "scalar dN = INV_TAU * (g_v - N);"

        # --- dq/dt, du/dt ---
        if self.config.mode != 'Class2':
            sim_code += """
            const scalar h_v = (v_curr < RH) ? fma(A_HN, (v_curr - B_HN) * (v_curr - B_HN), C_HN)
                                             : fma(A_HP, (v_curr - B_HP) * (v_curr - B_HP), C_HP);
            scalar dQ = EPSQ_OVER_TAU * (h_v - Q);
            """
        
        if self.config.mode in ['LTS', 'IB', 'PB']:
            sim_code += "scalar dU = EPSU_OVER_TAU * (v_curr - V0 - ALPU * U);"

        # オイラー法による更新 (GeNN の DT を秒単位に変換して使用)
        dt_sec = self.dt / 1000.0
        sim_code += f"const scalar dt_sec = (scalar){dt_sec};"
        sim_code += "V += dV * dt_sec; N += dN * dt_sec;"
        if self.config.mode != 'Class2': sim_code += "Q += dQ * dt_sec;"
        if self.config.mode in ['LTS', 'IB', 'PB']: sim_code += "U += dU * dt_sec;"

        return pygenn.create_neuron_model(
            f"PQN_Float_{self.config.mode}",
            params=list(self._params.keys()),
            vars=[("V", "scalar"), ("N", "scalar"), ("Q", "scalar"), ("U", "scalar"), ("Iext", "scalar")],
            sim_code=sim_code,
            threshold_condition_code="V >= V_THRESH"
        )

    @property
    def params(self): return self._params

    @property
    def initial_vars(self): return self._init_vars