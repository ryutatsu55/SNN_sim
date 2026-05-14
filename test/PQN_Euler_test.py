import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# プロジェクトルートにパスを通す
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
from models.neurons.pqn_origin import PQNengine


class PQNFloatEngine:
    def __init__(self, mode='RSexci'):
        self.orig_engine = PQNengine(mode)
        self.p = self.orig_engine.PARAM
        self.dt = self.p['dt']
        
        # 初期状態変数を浮動小数点に変換
        scale = 2 ** self.orig_engine.BIT_WIDTH_FRACTIONAL
        self.v = self.orig_engine.state_variable_v / scale
        self.n = self.orig_engine.state_variable_n / scale
        self.q = self.orig_engine.state_variable_q / scale

    def update(self, I_stim):
        p = self.p
        v = self.v
        
        if v < 0:
            fv = p['afn'] * (v - p['bfn'])**2 + p['cfn']
        else:
            fv = p['afp'] * (v - p['bfp'])**2 + p['cfp']

        if v < p['rg']:
            gv = p['agn'] * (v - p['bgn'])**2 + p['cgn']
        else:
            gv = p['agp'] * (v - p['bgp'])**2 + p['cgp']

        if v < p['rh']:
            hv = p['ahn'] * (v - p['bhn'])**2 + p['chn']
        else:
            hv = p['ahp'] * (v - p['bhp'])**2 + p['chp']

        dv = (p['phi'] / p['tau']) * (fv - self.n - self.q + p['I0'] + p['k'] * I_stim) * self.dt
        dn = (1 / p['tau']) * (gv - self.n) * self.dt
        dq = (p['epsq'] / p['tau']) * (hv - self.q) * self.dt

        self.v += dv
        self.n += dn
        self.q += dq

        return self.v

def run_simulation():
    # 1. シミュレーション条件の設定
    dt = 0.0001
    T_total = 2.0
    steps = int(T_total / dt)
    
    # 2. 事前確保によるデータ構造の最適化
    time = np.linspace(0, T_total, steps, endpoint=False)
    v_history = np.zeros(steps)
    I_history = np.zeros(steps)
    
    # 3. 刺激プロトコルの事前定義 (0.5s ~ 1.5sの間に0.09の電流を注入)
    stimulus_start = int(0.5 / dt)
    stimulus_end = int(1.5 / dt)
    I_history[stimulus_start:stimulus_end] = 0.16
    
    # 4. エンジンの初期化と計算ループ
    engine = PQNFloatEngine(mode='RSexci')
    
    for i in range(steps):
        v_history[i] = engine.update(I_history[i])
        
    # 5. 結果のプロット
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8, 4), sharex=True)
    
    ax1.plot(time, v_history, color='tab:blue')
    ax1.set_ylabel('v')
    ax1.set_xlim(0, T_total)
    
    ax2.plot(time, I_history, color='black')
    ax2.set_ylabel('I')
    ax2.set_xlabel('[s]')
    ax2.set_xlim(0, T_total)
    
    plt.tight_layout()
    plt.savefig("PQN_Euler_test.png")
    # plt.show()

if __name__ == '__main__':
    run_simulation()