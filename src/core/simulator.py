import pygenn
import numpy as np
from typing import Dict, Any, List, Tuple

class GeNNSimulator:
    def __init__(self, genn_model: pygenn.GeNNModel, config: Any, io_map: Dict[str, Any]):
        self.model = genn_model
        self.config = config
        self.io_map = io_map
        
        self.dt = self.config.simulation.dt
        # 1トライアルあたりの最大ステップ数 (事前メモリ確保用)
        self.max_timesteps = int(self.config.task.duration / self.dt)
        self.is_setup = False
        
        self.record_targets = self.io_map.get("outputs", {})
        self._recorded_vars_history = {}

    def setup(self):
        """GeNNモデルのコード生成、コンパイル、およびGPUへのロード"""
        print(f"=== [Simulator] Setup: Building model '{self.model.name}' ===")

        self.model.build()
        
        # 【重要】GPU上に記録用バッファを事前確保してロード
        self.model.load(num_recording_timesteps=self.max_timesteps)
        self.is_setup = True
        
        print(f"  [Simulator] Setup complete. Allocated buffer for {self.max_timesteps} steps.")

    def run(self, inputs: List[Tuple[Dict[str, np.ndarray], int]]) -> Dict[str, Dict[str, Any]]:
        """1トライアル分のシミュレーションを実行する"""
        if not self.is_setup:
            raise RuntimeError("Simulator must be setup before running.")

        self._recorded_vars_history = {
            pop_name: {var_name: [] for var_name in out_info.get("record_vars", []) if var_name != "spikes"}
            for pop_name, out_info in self.record_targets.items()
        }

        total_steps = sum(steps for _, steps in inputs)
        if total_steps > self.max_timesteps:
            raise ValueError(f"Input steps ({total_steps}) exceed allocated buffer ({self.max_timesteps}).")

        print(f"=== [Simulator] Running {total_steps * self.dt}ms trial ===")

        for update_dict, duration_steps in inputs:
            # デバイス(GPU)上の変数を更新
            for pop_name, data in update_dict.items():
                pop = self.model.neuron_populations[pop_name]
                target_var = self.io_map["inputs"][pop_name]["target_var"]
                
                pop.vars[target_var].view[:] = data
                pop.vars[target_var].push_to_device()

            # 変数を維持したまま指定ステップ数だけGPU上で計算を進める
            for _ in range(duration_steps):
                self.model.step_time()
                for pop_name, var_dict in self._recorded_vars_history.items():
                    pop = self.model.neuron_populations[pop_name]
                    for var_name, history_list in var_dict.items():
                        # GPUから値をプル
                        pop.vars[var_name].pull_from_device()
                        # viewのコピーを履歴に追加 (ニューロン数のサイズの1D配列)
                        history_list.append(np.copy(pop.vars[var_name].view))

        return self._gather_results()

    def _gather_results(self) -> Dict[str, Dict[str, Any]]:
        """デバイスから記録結果を一括で引き出す"""
        if getattr(self.model, "_recording_in_use", False):
            self.model.pull_recording_buffers_from_device()
        results = {}

        for pop_name, out_info in self.record_targets.items():
            pop = self.model.neuron_populations[pop_name]
            pop_results = {}
            vars_to_record = out_info.get("record_vars")
            
            if "spikes" in vars_to_record:
                spike_times, spike_ids = pop.spike_recording_data
                pop_results["spikes"] = {
                    "times": spike_times.copy(),
                    "ids": spike_ids.copy()
                }

            for var_name in vars_to_record:
                if var_name != "spikes":
                    history_list = self._recorded_vars_history[pop_name][var_name]
                    # [time_steps, num_neurons] の形状にする
                    pop_results[var_name] = np.stack(history_list, axis=0) if history_list else np.array([])

            results[pop_name] = pop_results

        return results

    def reset(self):
        """
        次のトライアルに向けてシミュレータの状態をリセットする。
        これを行わないと、時間が前回の続きから進み、メモリオーバーフローを引き起こす。
        """
        # 1. ネットワーク時間を0に戻す (これにより記録バッファのインデックスが先頭に戻る)
        # self.model.t = 0.0
        self.model.timestep = 0

        # 2. 変数(膜電位 V など)を初期状態に戻してGPUへ再転送
        # (NetworkBuilderで初期化された際の値をホストからデバイスへ押し込む)
        for pop in self.model.neuron_populations.values():
            for var_name, var in pop.vars.items():
                var.push_to_device()
                
        for pop in self.model.synapse_populations.values():
            for var_name, var in pop.vars.items():
                var.push_to_device()
                
        # ※ GeNN 5.x においてスパイク等の内部バッファポインタを明示的にクリアするAPI
        # （基本的には pull_recording_buffers_from_device を呼んでおけば問題ありませんが、
        # 念のため PyGeNN の reinit メソッド等で状態をクリーンにします）
        # self.model.reinit() # 必要に応じてコメントアウト解除
        print("  [Simulator] Network time and variables reset to 0.")