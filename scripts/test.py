import os
import sys

# プロジェクトルートにパスを通す
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.registry import DATA_LOADERS
from src.core.config_manager import ConfigManager
from src.core.NetworkBuilder import NetworkBuilder
from src.core.simulator import GeNNSimulator  # クラス名変更に対応
# from src.models.readouts.ridge_reg import RidgeReadout
import src.utils.visualize.visualize as visualize

# --- プラグイン(モデル)の登録トリガー ---
# ここでインポートすることで、@register デコレータが実行されレジストリに登録される
import src.models.neurons.PQN_float
import src.models.neurons.PQN_int
import src.models.network.space
import src.models.network.connectors
import src.models.network.weights
import src.models.network.delays
import src.data.test_data
# import src.models.neurons.lif  

TASK_NAME = "pqn_test"

def main():
    print("=== SNN_sim Test Pipeline Started ===")

    # 1. 設定の読み込み
    config_src = "test.yaml"
    print(f"Loading config from {config_src}...")
    manager = ConfigManager(config_src, "pqn_test") 
    config = manager.resolve()

    # 2. ネットワークの構築 (DataLoaderより先に実行して io_map を生成する)
    print("Building Network with GeNN...")
    builder = NetworkBuilder(config)
    genn_model, io_map = builder.build()
    
    # 3. データの準備 (io_mapを渡してグローバル→ローカルの変換ルールを教える)
    print("Preparing Input Data...")
    # config.data から使用するローダー名を取得（例: "input_type"）
    data_loader_class = DATA_LOADERS.get(TASK_NAME)
    
    if data_loader_class is None:
        raise ValueError(f"DataLoader '{TASK_NAME}' not found in registry.")
        
    data_loader = data_loader_class(config, io_map)
    
    # 4. シミュレーターの初期化とビルド
    print("Initializing Simulator...")
    sim = GeNNSimulator(genn_model, config, io_map)
    sim.setup() # GeNNのコンパイルとGPUメモリ確保
    
    # 5. シミュレーション実行 (トライアルごとのループ)
    print("Running Simulation Trials...")
    
    # 結果保存用のコンテナ (実験スクリプトで柔軟に取捨選択する想定)
    all_results = []
    
    for trial_idx, (trial_inputs, meta) in enumerate(data_loader.generate()):
        print(f"  --- Trial {trial_idx + 1} ---")
        
        # 1トライアル分を回して結果を引き出す
        trial_results = sim.run(trial_inputs)
        
        # ★ ここで必要なデータだけを保存する (柔軟なロギング)
        # 例: all_results.append((trial_results, meta))
        
        # 次のトライアルに向けて、ネットワークの時間と変数を初期化
        sim.reset()
        
    print("=== Simulation Complete! ===")
    
    # 6. Readout (学習)
    # print("Training Readout layer...")
    # readout = RidgeReadout()
    # weights = readout.fit(all_results, target_data)

    # 7. 評価と可視化
    I_in = data_loader.reconstruct(trial_inputs, target_pop=meta["target_pop"])
    visualize.PQN_test(trial_results["Layer_Exc"]["V"][:,0], I_in[:,0], config)

if __name__ == "__main__":
    main()