import pygenn
import numpy as np
from src.models.neurons.PQN_origin import PQNengine

def get_pqn_3var_class():
    # GeNN 5方式のカスタムニューロン定義 (関数名が create_neuron_model に変更されています)
    return pygenn.create_neuron_model(
        "PQN_3Var",
        params=[
            "PHI", "TAU", "I0", "K",
            "A_FN", "B_FN", "C_FN", "A_FP", "B_FP", "C_FP",
            "A_GN", "B_GN", "C_GN", "A_GP", "B_GP", "C_GP", "R_G",
            "A_HN", "B_HN", "C_HN", "A_HP", "B_HP", "C_HP", "R_H",
            "EPS_Q", "V_THRESH"
        ],
        vars=[
            ("V", "scalar"), 
            ("N", "scalar"), 
            ("Q", "scalar"),
            ("Iext", "scalar") # テスト用の外部入力電流の変数を追加
        ],
        sim_code="""
            // ブランチレス計算 (GeNN 5方式: $() を外す)
            const scalar a_f = (V >= 0.0) ? A_FP : A_FN;
            const scalar b_f = (V >= 0.0) ? B_FP : B_FN;
            const scalar c_f = (V >= 0.0) ? C_FP : C_FN;
            const scalar v_minus_b = V - b_f;
            const scalar f_v = fma(a_f * v_minus_b, v_minus_b, c_f);

            const scalar a_g = (V >= R_G) ? A_GP : A_GN;
            const scalar b_g = (V >= R_G) ? B_GP : B_GN;
            const scalar c_g = (V >= R_G) ? C_GP : C_GN;
            const scalar v_minus_bg = V - b_g;
            const scalar g_v = fma(a_g * v_minus_bg, v_minus_bg, c_g);

            const scalar a_h = (V >= R_H) ? A_HP : A_HN;
            const scalar b_h = (V >= R_H) ? B_HP : B_HN;
            const scalar c_h = (V >= R_H) ? C_HP : C_HN;
            const scalar v_minus_bh = V - b_h;
            const scalar h_v = fma(a_h * v_minus_bh, v_minus_bh, c_h);

            const scalar dt_sec = dt / 1000.0;

            const scalar dV_dt = (PHI / TAU) * (f_v - N - Q + I0 + K * Iext);
            const scalar dN_dt = (1.0 / TAU) * (g_v - N);
            const scalar dQ_dt = (EPS_Q / TAU) * (h_v - Q);

            V += dV_dt * dt_sec;
            N += dN_dt * dt_sec;
            Q += dQ_dt * dt_sec;
        """,
        threshold_condition_code="V >= V_THRESH"
    )

def get_genn_pqn_params(mode: str):
    PQN = PQNengine(mode=mode)
    p = PQN.PARAM
    # GeNN 5 向けにパラメータ名をマッピング
    params = {
        "PHI": float(p['phi']), "TAU": float(p['tau']), "I0": float(p['I0']), "K": float(p['k']),
        "A_FN": float(p['afn']), "B_FN": float(p['bfn']), "C_FN": float(p['cfn']),
        "A_FP": float(p['afp']), "B_FP": float(p['bfp']), "C_FP": float(p['cfp']),
        "A_GN": float(p['agn']), "B_GN": float(p['bgn']), "C_GN": float(p['cgn']),
        "A_GP": float(p['agp']), "B_GP": float(p['bgp']), "C_GP": float(p['cgp']), "R_G": float(p['rg']),
        "A_HN": float(p['ahn']), "B_HN": float(p['bhn']), "C_HN": float(p['chn']),
        "A_HP": float(p['ahp']), "B_HP": float(p['bhp']), "C_HP": float(p['chp']), "R_H": float(p['rh']),
        "EPS_Q": float(p['epsq']), "V_THRESH": float(p.get('v_thresh', 4.0))
    }

    def calculate_resting_state(p_dict):
        def f(v):
            a = p_dict['A_FP'] if v >= 0.0 else p_dict['A_FN']
            b = p_dict['B_FP'] if v >= 0.0 else p_dict['B_FN']
            c = p_dict['C_FP'] if v >= 0.0 else p_dict['C_FN']
            return a * (v - b)**2 + c

        def g(v):
            a = p_dict['A_GP'] if v >= p_dict['R_G'] else p_dict['A_GN']
            b = p_dict['B_GP'] if v >= p_dict['R_G'] else p_dict['B_GN']
            c = p_dict['C_GP'] if v >= p_dict['R_G'] else p_dict['C_GN']
            return a * (v - b)**2 + c

        def h(v):
            a = p_dict['A_HP'] if v >= p_dict['R_H'] else p_dict['A_HN']
            b = p_dict['B_HP'] if v >= p_dict['R_H'] else p_dict['B_HN']
            c = p_dict['C_HP'] if v >= p_dict['R_H'] else p_dict['C_HN']
            return a * (v - b)**2 + c

        # 釣り合いの式: dV/dt = 0 となる条件
        def F(v):
            return f(v) - g(v) - h(v) + p_dict['I0']

        # 二分法 (Bisection Method) によるゼロ点探索
        v_min, v_max = -20.0, 0.0 # 通常、静止膜電位は負の領域に存在する
        
        for _ in range(100):
            v_mid = (v_min + v_max) / 2.0
            if F(v_min) * F(v_mid) <= 0:
                v_max = v_mid
            else:
                v_min = v_mid
                
        v_rest = (v_min + v_max) / 2.0
        n_rest = g(v_rest)
        q_rest = h(v_rest)
        
        return v_rest, n_rest, q_rest

    # 動的に計算した値を初期値として設定
    v0, n0, q0 = calculate_resting_state(params)
    
    init_vars = {
        "V": v0,
        "N": n0,
        "Q": q0,
        "Iext": 0.0
    }

    return params, init_vars