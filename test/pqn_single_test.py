import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# プロジェクトルートにパスを通す
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
from simulator_V_test import Simulator

def main():
    # 本来は ConfigManager が YAML から生成する辞書を模倣
    # PQN_origin.py の "RSexci" モードを指定し、定電流を与える
    mock_config = {
        "dt": 1.0,  # 0.1 ms
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
    I = np.zeros(number_of_iterations)
    I[int(number_of_iterations/4):int(number_of_iterations/4*3)] = 0.09
    
    # 実行して膜電位Vの履歴を取得
    v_trace = sim.run_test(tmax_s=tmax, input_current=I)

    # 結果の可視化
    time_axis = np.arange(len(v_trace)) * mock_config["dt"]
    
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, v_trace, label="Membrane Potential (V)")
    plt.title(f"PQN Neuron Dynamics Test (Mode: {mock_config['mode']})")
    plt.xlabel("Time [ms]")
    plt.ylabel("V")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("V_test.png")

if __name__ == "__main__":
    main()