import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import sys
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
from src.core.config_manager import ConfigManager
from src.models.neurons.pqn import PQNNeuron
from src.core.simulator import Simulator

def main():
    print("Initializing Test...")
    config_path = "configs/test.yaml"
    # config_path = root_path / "configs" / "test.yaml"
    config_mgr = ConfigManager(config_path)
    config_data = config_mgr.get_data()
    
    dt = config_data["network"]["dt"]
    num_neurons = config_data["network"]["num_neurons"]
    # 0: RSexci, 1: RSinhi, 2: FS, 3: LTS, 4: IB, 5: EB, 6: PB
    test_type_idx = 0 
    test_type_name = "RSexci"

    # 2. トポロジ(結合)データの準備 (今回は単一テストなので空の辞書)
    network_data = {} 
    
    # 3. モデルの準備
    # 0番目のニューロンを RSexci(0), 1番目を FS(2) に設定
    neuron_types = np.array([test_type_idx], dtype=np.uint8) 
    pqn_model = PQNNeuron(num_neurons=num_neurons, neuron_types=neuron_types)
    
    # シミュレータの準備
    sim = Simulator(config=config_data, network_data=network_data, neuron_model=pqn_model)


    # 1000ステップ (100ms) シミュレーションを実行
    tmax = 2    # [s]
    steps = int(tmax/dt)
    I_history = np.zeros((steps, num_neurons), dtype=np.float32)
    
    # 例：10ms 〜 90ms の間だけ 2.5 の電流を流す
    start_step = int(0.50 / dt)
    end_step = int(1.5 / dt)
    I_history[start_step:end_step, 0] = 0.09

    print(f"Running simulation for {steps} steps...")
    v_history = sim.run(steps=steps, I_history=I_history)

    # グラフの描画
    time_axis = np.arange(steps) * dt # ms単位

    fig = plt.figure(figsize=(8,4))
    spec = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace=0.1, height_ratios=[4, 1])
    ax0 = fig.add_subplot(spec[0])
    ax1 = fig.add_subplot(spec[1])
    ax0.plot(time_axis, v_history)
    ax0.set_xlim(0, tmax)
    ax0.set_ylabel("v")
    ax0.set_xticks([])
    ax1.plot(time_axis, I_history, color="black")
    ax1.set_xlim(0, tmax)
    ax1.set_xlabel("[s]")
    ax1.set_ylabel("I")
    
    plt.savefig("test_pqn_result.png")
    print("Saved result to test_pqn_result.png! Check the plot.")

if __name__ == "__main__":
    main()