import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# プロジェクトルートにパスを通す
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
from test.archive.simulator_V_test import Simulator

def main():
    # 本来は ConfigManager が YAML から生成する辞書を模倣
    # PQN_origin.py の "RSexci" モードを指定し、定電流を与える
    mock_config = {
        "dt": 0.1,  # 0.1 ms
        "mode": "RSexci",
        "params": {
            "I0": 0.0 # ベース電流の上書きテスト
        }
    }

    # シミュレータの初期化とGeNNコンパイル
    sim = Simulator(mock_config)

    # 1秒間(1.0s)、強めの電流を与えてシミュレーションを回す
    tmax = 2.0
    input_current = 0.15
    number_of_iterations = int(tmax / mock_config["dt"] * 1000)
    I_in = np.zeros(number_of_iterations)
    I_in[int(number_of_iterations/4):int(number_of_iterations/4*3)] = input_current
    
    # 実行して膜電位Vの履歴を取得
    v_trace = sim.run_test(tmax_s=tmax, input_current=I_in)

    # 結果の可視化
    time_axis = np.arange(len(v_trace)) * mock_config["dt"] / 1000.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8, 4), sharex=True)
    
    ax1.plot(time_axis, v_trace, color='tab:blue')
    ax1.set_ylabel('v')
    ax1.set_xlim(0, tmax)
    
    ax2.plot(time_axis, I_in, color='black')
    ax2.set_ylabel('I')
    ax2.set_xlabel('[s]')
    ax2.set_xlim(0, tmax)
    
    plt.tight_layout()
    plt.savefig("V_test.png")

if __name__ == "__main__":
    main()