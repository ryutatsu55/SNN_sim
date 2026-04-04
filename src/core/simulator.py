import pygenn
import numpy as np

class Simulator:
    def __init__(self, genn_model: pygenn.GeNNModel, sim_config: dict, group_info: dict = None):
        self.model = genn_model
        self.config = sim_config
        self.dt = sim_config.get("dt", 0.1)
        self.duration = sim_config.get("duration", 100.0)
        self.timesteps = int(self.duration / self.dt)
        self.is_setup = False
        
        # NetworkBuilderから渡される、グループ名とグローバルインデックスの対応辞書
        self.group_info = group_info or {}

    def setup(self):
        """GeNNのコード生成、コンパイル、およびGPUへのロード"""
        print(f"  [Simulator] Building and Loading Model: {self.model.name}...")

        # スパイク記録の有効化
        for pop in self.model.neuron_populations.values():
            pop.spike_recording_enabled = True

        self.model.build()
        self.model.load(num_recording_timesteps=self.timesteps)
            
        self.is_setup = True
        print("  [Simulator] Setup complete.")

    def run(self, input_data: np.ndarray = None):
        """
        シミュレーションを実行し、毎ステップの膜電位 V を取得するテスト用メソッド。
        
        Parameters:
        -----------
        input_data : np.ndarray
            時間変化する入力の場合は (timesteps, total_neurons)
            一定の電流を流し続ける場合は (total_neurons,) の配列
        """
        if not self.is_setup:
            raise RuntimeError("Simulator must be setup before running.")

        print(f"  [Simulator] Starting {self.duration}ms simulation ({self.timesteps} steps)...")

        # 記録用バッファの準備
        results = {
            "spikes": {},
            "v_history": {} # グループごとに毎ステップのVを記録
        }
        
        for name in self.model.neuron_populations:
            results["spikes"][name] = []
            results["v_history"][name] = []

        for t in range(self.timesteps):
            # --- 1. 外部入力 (Iext) の更新と転送 ---
            # 毎ステップ入力が変化する場合、または最初のステップのみ転送
            if input_data is not None:
                current_input = input_data[t]
                
                for name, pop in self.model.neuron_populations.items():
                    # グローバル配列から、このグループに属するニューロンの入力だけを抽出
                    global_idx = self.group_info[name]["global_indices"]
                    group_input = current_input[global_idx]

                    # デバイス（GPU/CPU）へ転送
                    pop.vars["Iext"].view[:] = group_input
                    pop.vars["Iext"].push_to_device()

            # --- 2. シミュレーションを1ステップ進める ---
            self.model.step_time()
            
            # --- 3. 毎ステップの膜電位 (V) を引き出す ---
            for name, pop in self.model.neuron_populations.items():
                pop.vars["V"].pull_from_device()
                # .copy() をしないと、PyGeNNの参照バッファが毎ステップ上書きされてしまうため必須
                results["v_history"][name].append(pop.vars["V"].view.copy())

        # --- 4. スパイクデータの回収 (ループ終了後に一括) ---
        self.model.pull_recording_buffers_from_device()
        for name, pop in self.model.neuron_populations.items():
            results["spikes"][name] = pop.spike_recording_data
            # Vの履歴をnumpy配列 (timesteps, num_neurons) に変換して扱いやすくする
            results["v_history"][name] = np.array(results["v_history"][name])

        return results
    