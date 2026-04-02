import os
import sys

# プロジェクトルートにパスを通す
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.registry import DATA_LOADERS
from src.core.config_manager import ConfigManager
from src.core.NetworkBuilder import NetworkBuilder
# from src.core.simulator import Simulator
# from src.data.spatial_loader import DataLoader
# from src.models.readouts.ridge_reg import RidgeReadout
# from src.utils.visualize import plot_raster

# --- プラグイン(モデル)の登録トリガー ---
# ここでインポートすることで、@register デコレータが実行されレジストリに登録される
import src.models.neurons.PQN_float
import src.models.neurons.PQN_int
# import src.models.neurons.lif  # 将来追加するモデルもここに書く

def main():
    print("=== SNN_sim Test Pipeline Started ===")

    # 1. 設定の読み込み
    config_path = "configs/test.yaml"
    print(f"Loading config from {config_path}...")
    config = ConfigManager(config_path).get_config()

    # 2. データの準備 (モックアップ)
    print("Preparing Input Data...")
    # data_loader = DataLoader(config['data'])
    # input_data = data_loader.get_inputs()
    
    # 3. ネットワークの構築
    print("Building Network with GeNN...")
    builder = NetworkBuilder(config)
    genn_model = builder.build()
    
    # 4. シミュレーターの実行 (今回はBuilderのテストまでなのでコメントアウト)
    # print("Initializing Simulator and Running...")
    # sim = Simulator(genn_model, config['simulation'])
    # sim.setup() # GeNNのコンパイルとロード
    # results = sim.run(input_data)
    
    # 5. Readout (学習)
    # print("Training Readout layer...")
    # readout = RidgeReadout()
    # weights = readout.fit(results['spikes'], target_data)

    # 6. 評価と可視化
    # print("Evaluating and Visualizing...")
    # plot_raster(results['spikes'])
    
    print("=== Network Built Successfully ===")
    print(f"Model Name: {genn_model.name}")
    print(f"Neurons Groups: {[g.name for g in genn_model.neuron_groups]}")

if __name__ == "__main__":
    main()