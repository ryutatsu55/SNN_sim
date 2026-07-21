import os
import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path

# プロジェクトルートにパスを通す
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.registry import DATA_LOADERS
from src.core.config_manager import ConfigManager
from src.core.NetworkBuilder import NetworkBuilder
from src.core.output_manager import create_run_output_dir
from src.core.simulator import GeNNSimulator  # クラス名変更に対応
# from src.models.readouts.ridge_reg import RidgeReadout
import src.utils.visualize as visualize

# --- プラグイン(モデル)の登録トリガー ---
# ここでインポートすることで、@register デコレータが実行されレジストリに登録される
import src.models.neurons.pqn_float
import src.models.neurons.pqn_int
import src.models.neurons.akita_escape_lif
import src.models.neurons.akita_escape_lif_physical
import src.models.neurons.lif
import src.models.network.space
import src.models.network.connectors
import src.models.network.weights
import src.models.network.delays
import src.models.synapses.standard_models
import src.models.synapses.custom
import src.models.plasticity.custom_Akita
import src.models.plasticity.standard_models
import src.data.test_data
# import src.models.neurons.lif  

TASK_NAME = "pqn_test"

def main():
    print("=== SNN_sim Test Pipeline Started ===")

    # 1. 設定の読み込み
    config_src = "configs/test.yaml"
    print(f"Loading config from {config_src}...")
    manager = ConfigManager() 
    config = manager.resolve(config_src, TASK_NAME)
    output_dir = create_run_output_dir(TASK_NAME)
    print(f"Output directory: {output_dir}")

    # 2. ネットワークの構築 (DataLoaderより先に実行して layout を生成する)
    print("Building Network with GeNN...")
    builder = NetworkBuilder(config)
    genn_model, layout = builder.build(rec_spike=True)

    # 3. データの準備 (layoutを渡してグローバル→ローカルの変換ルールを教える)
    print("Preparing Input Data...")
    # config.data から使用するローダー名を取得（例: "input_type"）
    data_loader_class = DATA_LOADERS.get(TASK_NAME)
    
    if data_loader_class is None:
        raise ValueError(f"DataLoader '{TASK_NAME}' not found in registry.")
        
    data_loader = data_loader_class(config, layout)
    
    # 4. シミュレーターの初期化とビルド
    print("Initializing Simulator...")
    sim = GeNNSimulator(genn_model, config, builder)
    sim.setup() # GeNNのコンパイルとGPUメモリ確保
    
    # 5. シミュレーション実行 (トライアルごとのループ)
    print("Running Simulation Trials...")
    
    # 結果保存用のコンテナ (実験スクリプトで柔軟に取捨選択する想定)
    results = np.zeros((data_loader.total_steps, builder.total_neurons))  # 例: 全ニューロンの膜電位を保存する場合
    I_in = np.zeros((data_loader.total_steps, builder.total_neurons))
    
    for trial_idx, (trial_inputs, meta) in enumerate(data_loader.generate()):
        print(f"  --- Trial {trial_idx + 1} ---")
        
        # --- 新しい制御フロー ---
        step=0
        for inputs, duration_steps in trial_inputs:
            # 1. スパイク入力データをGPUに転送
            # ※連続値(Iextなど)をスナップショットで渡したい場合は sim.push() を併用
            sim.push(inputs, target_var="Iext")
            I_in[step:step+duration_steps] = inputs
            for i in range(duration_steps):
                sim.step()
                results[step,:] = sim.pull("V")
                # I_in[step,:] = sim.pull("Isyn")
                step += 1

        # 3. デバイスから記録バッファを一括で引き出す
        trial_results = sim.get_global_spikes()

        indices = np.where(trial_results["ids"] == 0)[0]
        print(trial_results["times"][indices])
        
        # 4. 次のトライアルに向けて、ネットワーク時間と変数を初期化
        sim.reset()
        
    print("=== Simulation Complete! ===")
    
    manager.save_resolved(config, save_dir=output_dir)

    # 7. 評価と可視化
    # I_in[:] = config.neurons["Layer_Exc"].Ioffset

    visualize.neuron_test(
        results,
        I_in,
        trial_results["times"],
        trial_results["ids"], 
        config,
        save_path=output_dir
    )

    visualize.network(
        weights=builder.global_weights, 
        coords=builder.global_coords, 
        config=config,
        save_path=output_dir
    )

    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
