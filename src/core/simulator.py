import pygenn
import numpy as np
from typing import Dict, Any, List, Tuple

class GeNNSimulator:
    def __init__(self, genn_model: pygenn.GeNNModel, config: Any, group_info: Dict[str, Any]):
        self.model = genn_model
        self.config = config
        self.group_info = group_info

        self.dt = self.config.simulation.dt
        # 1トライアルあたりの最大ステップ数 (事前メモリ確保用)
        self.max_timesteps = int(self.config.task.duration / self.dt)
        self.total_neurons = sum(info["num"] for _, info in self.group_info.items())
        
        self.is_setup = False

    def setup(self):
        """GeNNモデルのコード生成、コンパイル、およびGPUへのロード"""
        print(f"=== [Simulator] Setup: Building model '{self.model.name}' ===")

        self.model.build()
        
        # 【重要】GPU上に記録用バッファを事前確保してロード
        self.model.load(num_recording_timesteps=self.max_timesteps)

        self.initial_states = {}
        for pop_name, pop in self.model.neuron_populations.items():
            self.initial_states[pop_name] = {}
            for var_name, var in pop.vars.items():
                self.initial_states[pop_name][var_name] = np.copy(var.view)

        self.is_setup = True
        
        print(f"  [Simulator] Setup complete. Allocated buffer for {self.max_timesteps} steps.")

    # def flush_recording(self):
    #     """
    #     GPU上の記録バッファをCPUに転送し、内部カウンタをリセットして『捨てる』
    #     これにより、次のステップからバッファの先頭に上書きされるようになる
    #     """
    #     # GPUからデータを引き上げる（これを行わないとGPU側で上書きが起きるか停止する）
    #     self.model.pull_recording_buffers_from_device()
        
    #     # 重要：ホスト（CPU）側の記録カウンタを0リセットする
    #     # これにより、これまでにpullしたスパイクデータがクリアされる
    #     self.model.reset_recording_counters()
    #     # print("  [Simulator] Recording buffer flushed and reset.")

    # def load_input_spikes(self, spike_data: Dict[str, Dict[str, np.ndarray]]):
    #     """
    #     [トライアル毎] SpikeSourceArrayに入力データをセットし、GPUへ転送する
    #     """
    #     for pop_name, data in spike_data.items():
    #         pop = self.model.neuron_populations[pop_name]
    #         spike_times = data["times"]
    #         spike_ids = data["ids"]
    #         # SpikeSourceArray特有の変数を更新 (開始/終了インデックスなど)
    #         # ※定義したSnippetの仕様によって変数名は変わりますが、基本形はこれです
    #         pop.vars["startSpike"].view[:] = ...
    #         pop.vars["endSpike"].view[:] = ...
            
    #         # タイムスタンプとIDの配列をカスタム配列（Extra Global Params等）に流し込む
    #         self.model.custom_input_arrays["spike_times"].view[:] = spike_times
    #         self.model.custom_input_arrays["spike_ids"].view[:] = spike_ids
            
    #         # GPUへ転送
    #         pop.push_var_to_device("startSpike")
    #         pop.push_var_to_device("endSpike")
    #         self.model.custom_input_arrays["spike_times"].push_to_device()
    #         self.model.custom_input_arrays["spike_ids"].push_to_device()

    def push(self, global_data: np.ndarray, target_var: str = "Iext"):
        """
        外部からの入力(形状: total_neurons)を、各Populationの対象変数に流し込む
        """
        local_data_dict = self._split_global_to_local(global_data)
        for pop_name, data in local_data_dict.items():
            pop = self.model.neuron_populations[pop_name]
            
            pop.vars[target_var].view[:] = data
            pop.vars[target_var].push_to_device()

    def step(self, duration_steps: int=1):
        """現在の変数の状態を維持したまま、指定ステップ数だけ時間を進める"""
        for _ in range(duration_steps):
            self.model.step_time()

    def pull(self, var_name: str) -> np.ndarray:
        """
        ネットワーク全体の対象変数の現在状態を、(total_neurons,) の形状で引き上げる
        """
        local_data_dict = {}
        for pop_name in self.group_info.keys():
            pop = self.model.neuron_populations[pop_name]
            
            # GPUから現在の値をプルして辞書に格納
            pop.vars[var_name].pull_from_device()
            local_data_dict[pop_name] = np.copy(pop.vars[var_name].view)
            
        # グローバル配列に再構築して返す
        return self._merge_local_to_global(local_data_dict)

    def get_global_spikes(self) -> Dict[str, np.ndarray]:
        """
        全Populationのスパイク記録を収集し、グローバルIDに変換して時間順にソートした結果を返す
        """
        if getattr(self.model, "_recording_in_use", False):
            self.model.pull_recording_buffers_from_device()

        all_times = []
        all_global_ids = []

        for pop_name, info in self.group_info.items():
            # GeNNからローカルデータを直接参照
            times, local_ids = self.model.neuron_populations[pop_name].spike_recording_data[0]
            
            if len(times) > 0:
                all_times.append(times)
                # ループ内で直接グローバルインデックスにマッピング
                # info["global_indices"] は以前に作成したnumpy配列を想定
                all_global_ids.append(info["global_indices"][local_ids])

        # スパイクが0件の場合の早期リターン
        if not all_times:
            return {"times": np.array([], dtype=np.float32), "ids": np.array([], dtype=np.int32)}

        # 結合とソート（複数Populationの混在を時間軸で整列）
        flat_times = np.concatenate(all_times)
        flat_ids = np.concatenate(all_global_ids)
        sort_idx = np.argsort(flat_times)

        return {
            "times": flat_times[sort_idx],
            "ids": flat_ids[sort_idx]
        }

    def reset(self):
        """
        次のトライアルに向けてシミュレータの状態をリセット
        """
        self.model.timestep = 0
        # self.model.t = 0.0

        # バックアップしておいた初期値で view を上書きしてから GPU へ送る
        for pop_name, pop in self.model.neuron_populations.items():
            for var_name, var in pop.vars.items():
                var.view[:] = self.initial_states[pop_name][var_name]
                var.push_to_device()
                
        print("  [Simulator] Network time and variables safely reset to initial states.")

    def _split_global_to_local(self, global_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        [形状: (total_neurons,)] のグローバル配列を Population毎に分割する
        """
        if len(global_data) != self.total_neurons:
            raise ValueError(f"Input data size {len(global_data)} does not match total neurons {self.total_neurons}")

        local_dict = {}
        for pop_name, info in self.group_info.items():
            global_idx = info["global_indices"]
            # fancy indexingによる切り出し
            local_dict[pop_name] = global_data[global_idx]
        return local_dict
    
    def _merge_local_to_global(self, local_dict: Dict[str, np.ndarray], dtype=np.float32) -> np.ndarray:
        """
        Population毎の配列を [形状: (total_neurons,)] のグローバル配列に結合する
        """
        global_data = np.zeros(self.total_neurons, dtype=dtype)
        for pop_name, local_data in local_dict.items():
            global_idx = self.group_info[pop_name]["global_indices"]
            global_data[global_idx] = local_data
        return global_data
    