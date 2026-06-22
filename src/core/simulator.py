import pygenn
import numpy as np
from typing import Dict, Any, List, Tuple
from src.core.NetworkBuilder import NetworkBuilder

class GeNNSimulator:
    def __init__(self, genn_model: pygenn.GeNNModel, config: Any, builder: NetworkBuilder):
        self.model = genn_model
        self.config = config
        self.group_info = builder.group_info
        self.builder = builder

        self.dt = self.config.simulation.dt
        # スパイク記録バッファ長。長時間実験では記録窓ぶんだけ確保する。
        recording_buffer_ms = getattr(self.config.task, "record_buffer_ms", None)
        if recording_buffer_ms is None:
            recording_buffer_ms = getattr(self.config.task, "record_window_ms", self.config.task.duration)
        self.max_timesteps = int(recording_buffer_ms / self.dt)
        self.total_neurons = sum(info["num"] for _, info in self.group_info.items())
        
        self.is_setup = False

    def setup(self):
        """GeNNモデルのコード生成、コンパイル、およびGPUへのロード"""
        print(f"=== [Simulator] Setup: Building model '{self.model.name}' ===")

        self.model.build()
        
        # 【重要】GPU上に記録用バッファを事前確保してロード
        # この時点で各変数の初期値がホスト側のメモリ(view)に展開されます
        self.model.load(num_recording_timesteps=self.max_timesteps)

        # 全ての初期状態を階層的に保存するための辞書を初期化
        self.initial_states = {
            'neurons': {},
            'synapses': {},
            'current_sources': {}
        }

        # 1. ニューロンの初期状態保存
        # GPU で初期化された値を CPU に pull してから保存する
        for pop_name, pop in self.model.neuron_populations.items():
            self.initial_states['neurons'][pop_name] = {}
            for var_name, var in pop.vars.items():
                var.pull_from_device()
                self.initial_states['neurons'][pop_name][var_name] = np.copy(var.view)

        # 2. シナプスの初期状態保存 (vars, pre_vars, post_vars)
        # STDPのトレース(x)や発火時刻(t_last_pre)をリセットするために必須
        for pop_name, pop in self.model.synapse_populations.items():
            self.initial_states['synapses'][pop_name] = {
                'vars': {},
                'pre_vars': {},
                'post_vars': {}
            }
            # 接続ごとの変数 (重みgなど)
            for var_name, var in pop.vars.items():
                var.pull_from_device()
                self.initial_states['synapses'][pop_name]['vars'][var_name] = np.copy(var.values)

            # Preニューロンごとの変数 (STDPトレースなど)
            for var_name, var in pop.pre_vars.items():
                var.pull_from_device()
                self.initial_states['synapses'][pop_name]['pre_vars'][var_name] = np.copy(var.values)

            # Postニューロンごとの変数
            for var_name, var in pop.post_vars.items():
                var.pull_from_device()
                self.initial_states['synapses'][pop_name]['post_vars'][var_name] = np.copy(var.values)

        # 3. カレントソース（入力源）の初期状態保存
        for cs_name, cs in self.model.current_sources.items():
            self.initial_states['current_sources'][cs_name] = {}
            for var_name, var in cs.vars.items():
                var.pull_from_device()
                self.initial_states['current_sources'][cs_name][var_name] = np.copy(var.values)

        self.is_setup = True
        print("  [Simulator] Setup complete. All initial states backed up for multi-trial reset.")

    def flush_recording(self):
        """
        GPU上の記録バッファをCPUに転送し、内部カウンタをリセットして捨てる。
        長時間の自発活動で記録バッファを小さく保つために使う。
        """
        if getattr(self.model, "_recording_in_use", False):
            self.model.pull_recording_buffers_from_device()

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
        for pop_name in self.model.neuron_populations.keys():
            pop = self.model.neuron_populations[pop_name]
            
            # GPUから現在の値をプルして辞書に格納
            pop.vars[var_name].pull_from_device()
            local_data_dict[pop_name] = np.copy(pop.vars[var_name].view)
            
        # グローバル配列に再構築して返す
        return self._merge_local_to_global(local_data_dict)

    def pull_pre_var(self, var_name: str, dtype=np.float32) -> np.ndarray:
        """シナプス集団の pre-var をグローバル (total_neurons,) 配列で返す"""
        global_data = np.full(self.total_neurons, np.nan, dtype=dtype)
        for syn_pop_name, syn_pop in self.model.synapse_populations.items():
            if var_name not in syn_pop.pre_vars:
                continue
            syn_pop.pre_vars[var_name].pull_from_device()
            values = np.copy(syn_pop.pre_vars[var_name].values)
            src_name, _, _ = syn_pop_name.partition("_to_")
            global_data[self.group_info[src_name]["global_indices"]] = values
        return global_data

    def pull_synapse(self, var_name: str) -> np.ndarray:
        """
        ネットワーク全体のシナプス変数の現在状態を、
        (total_neurons, total_neurons) のグローバル行列の形状で引き上げる
        """
        # 1. 現在のグローバル行列の雛形を作成 (初期値0)
        global_matrix = np.zeros((self.total_neurons, self.total_neurons), dtype=np.float32)
        
        # 2. 全てのシナプスポピュレーションを走査
        for syn_pop_name, syn_pop in self.model.synapse_populations.items():
            # GPUから最新の値を引き上げる
            syn_pop.vars[var_name].pull_from_device()
            
            # このポピュレーションにおける各接続の現在の値 (flatな配列)
            current_values = syn_pop.vars[var_name].values
            
            # NetworkBuilder側で保持している「どのグローバル座標にマッピングするか」の情報を使用
            # syn_pop_name は "src_to_tgt" の形式であることを前提とする
            src_name, _, tgt_name = syn_pop_name.partition("_to_")
            
            src_indices = self.group_info[src_name]["global_indices"]
            tgt_indices = self.group_info[tgt_name]["global_indices"]
            
            # SPARSE接続のインデックスを取得 (NetworkBuilderの _build_synapses で指定したもの)
            # ※もしシミュレーター側で保持していない場合は、再計算が必要
            sub_mask = self.builder.global_mask[np.ix_(src_indices, tgt_indices)]
            local_src_idx, local_tgt_idx = np.where(sub_mask != 0)
            
            # グローバル行列の該当箇所に値を書き戻す
            global_indices_src = src_indices[local_src_idx]
            global_indices_tgt = tgt_indices[local_tgt_idx]
            
            global_matrix[global_indices_src, global_indices_tgt] = current_values
            
        return global_matrix

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
            return {"times": np.array([], dtype=np.float64), "ids": np.array([], dtype=np.int32)}

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
        次のトライアルに向けてシミュレータの状態（ニューロン、シナプス、入力源）を完全にリセット
        """
        self.model.timestep = 0
        # self.model.t = 0.0

        # 1. ニューロン変数のリセット (V, RefracTime など)
        for pop_name, pop in self.model.neuron_populations.items():
            if pop_name in self.initial_states['neurons']:
                for var_name, var in pop.vars.items():
                    var.view[:] = self.initial_states['neurons'][pop_name][var_name]
                    var.push_to_device()
            # sT (スパイク時刻) のリセット。timestep=0 で t が 0 に戻るのに sT だけ
            # 前試行の正値が残ると dt_pre_arrival = t - d*dt - st_pre が負になり
            # ロールバック条件を誤って満たしてしまうため -TIME_MAX に戻す。
            if pop.spike_times is not None:
                pop.spike_times.view[:] = -np.finfo(pop.spike_times.view.dtype).max
                pop.spike_times.push_to_device()

        # 2. シナプス変数のリセット (g, d, x, t_last_pre など)
        # STDPモデルなどの学習状態を初期化するために必須です
        for pop_name, pop in self.model.synapse_populations.items():
            if pop_name in self.initial_states['synapses']:
                # 接続ごとの変数 (g, d など)
                for var_name, var in pop.vars.items():
                    var.values = self.initial_states['synapses'][pop_name]['vars'][var_name]
                    var.push_to_device()
                
                # プレニューロンごとの変数 (STDPトレース x, t_last_pre など)
                for var_name, var in pop.pre_vars.items():
                    var.values = self.initial_states['synapses'][pop_name]['pre_vars'][var_name]
                    var.push_to_device()
                
                # ポストニューロンごとの変数 (もし定義されていれば)
                for var_name, var in pop.post_vars.items():
                    var.values = self.initial_states['synapses'][pop_name]['post_vars'][var_name]
                    var.push_to_device()

        # 3. カレントソース（入力源）変数のリセット
        for cs_name, cs in self.model.current_sources.items():
            if cs_name in self.initial_states['current_sources']:
                for var_name, var in cs.vars.items():
                    var.view[:] = self.initial_states['current_sources'][cs_name][var_name]
                    var.push_to_device()
                    
        print("  [Simulator] All network variables (Neurons, Synapses, Inputs) safely reset.")

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
    
