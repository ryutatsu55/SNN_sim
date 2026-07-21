import os
import pygenn
import numpy as np
from typing import Dict, Any, List, Tuple
from src.core.NetworkBuilder import NetworkBuilder

# pull_synapse が確保してよい密行列の上限。これを超えたら COO 版へ誘導する。
# (N=40000 では密行列だけで 6.4 GiB になり、ホストメモリが尽きる)
DENSE_PULL_LIMIT_BYTES = 2 * 1024 ** 3


class GeNNSimulator:
    def __init__(self, genn_model: pygenn.GeNNModel, config: Any, builder: NetworkBuilder):
        self.model = genn_model
        self.config = config
        self.layout = builder.layout
        self.builder = builder

        self.dt = self.config.simulation.dt
        # スパイク記録バッファ長。長時間実験では記録窓ぶんだけ確保する。
        recording_buffer_ms = getattr(self.config.task, "record_buffer_ms", None)
        if recording_buffer_ms is None:
            recording_buffer_ms = getattr(self.config.task, "record_window_ms", self.config.task.duration)
        self.max_timesteps = int(recording_buffer_ms / self.dt)
        self.total_neurons = self.layout.total_neurons
        
        self.is_setup = False

    def setup(self, backup_initial_states: bool = True):
        """GeNNモデルのコード生成、コンパイル、およびGPUへのロード

        Args:
            backup_initial_states: 全変数の初期状態を控えるか。`reset()` (複数トライアル実行)
                に必須だが、大規模ネットワークではシナプス変数の控えだけで GB 級のホストメモリを
                消費し、かつ pygenn の値取得が行ごとの Python ループなので setup が非常に遅い。
                単発の自発活動シミュレーションのように `reset()` を呼ばない用途では False にする。
        """
        print(f"=== [Simulator] Setup: Building model '{self.model.name}' ===")

        # コード生成先。builder.code_gen_dir が指定されていれば
        # <code_gen_dir>/<model_name>_CODE に集約する (プロジェクト直下への散乱を防ぐ)。
        build_path = getattr(self.builder, "code_gen_dir", None)
        if build_path:
            os.makedirs(build_path, exist_ok=True)
            print(f"  [Simulator] GeNN code dir: {build_path}/{self.model.name}_CODE")
            self.model.build(path_to_model=build_path)
        else:
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

        if not backup_initial_states:
            self.initial_states = None
            self.is_setup = True
            print("  [Simulator] Setup complete (initial states NOT backed up; reset() is unavailable).")
            return

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
            global_data[self.layout.global_indices(src_name)] = values
        return global_data

    def pull_synapse_flat(self, var_name: str) -> Dict[str, np.ndarray]:
        """シナプス変数を集団ごとの 1D 配列として引き上げる。

        各配列の並びは `builder.synapse_index[pop].local_src/local_tgt` と一致する
        (集団ローカルの (pre, post) 行優先ソート順)。密行列を作らないため、
        大規模ネットワークではこちらを使う。
        """
        values: Dict[str, np.ndarray] = {}
        for syn_pop_name, syn_pop in self.model.synapse_populations.items():
            syn_pop.vars[var_name].pull_from_device()
            values[syn_pop_name] = np.copy(syn_pop.vars[var_name].values)
        return values

    def pull_synapse_coo(self, var_name: str) -> Dict[str, Any]:
        """シナプス変数をグローバルID空間の COO 形式で引き上げる。

        Returns:
            row / col       : int32, グローバル pre/post ID
            data            : float32, 対応する変数値
            pair_names      : シナプス集団名 (連結順)
            pair_offsets    : 各集団の data 内での開始位置 (末尾に総数)
            shape           : (total_neurons, total_neurons)

        連結順は集団ごとの「ペア major」であり、グローバルにソートはしない
        (数千万要素のソートを記録のたびに払わないため)。
        """
        rows, cols, datas = [], [], []
        pair_names, pair_offsets = [], [0]

        for syn_pop_name, syn_pop in self.model.synapse_populations.items():
            index = self.builder.synapse_index.get(syn_pop_name)
            if index is None:
                raise KeyError(
                    f"synapse_index に '{syn_pop_name}' がありません。"
                    " NetworkBuilder.build() を経ずに構築されたモデルの可能性があります。"
                )
            syn_pop.vars[var_name].pull_from_device()
            values = np.asarray(syn_pop.vars[var_name].values, dtype=np.float32)
            if values.size != index.num_synapses:
                raise ValueError(
                    f"'{syn_pop_name}' の値数 {values.size} が記録済み接続数 "
                    f"{index.num_synapses} と一致しません。"
                )
            rows.append(index.global_src)
            cols.append(index.global_tgt)
            datas.append(values)
            pair_names.append(syn_pop_name)
            pair_offsets.append(pair_offsets[-1] + values.size)

        empty_i = np.array([], dtype=np.int32)
        empty_f = np.array([], dtype=np.float32)
        return {
            "row": np.concatenate(rows) if rows else empty_i,
            "col": np.concatenate(cols) if cols else empty_i,
            "data": np.concatenate(datas) if datas else empty_f,
            "pair_names": np.array(pair_names, dtype=object),
            "pair_offsets": np.array(pair_offsets, dtype=np.int64),
            "shape": np.array([self.total_neurons, self.total_neurons], dtype=np.int64),
        }

    def pull_synapse(self, var_name: str) -> np.ndarray:
        """
        ネットワーク全体のシナプス変数の現在状態を、
        (total_neurons, total_neurons) のグローバル行列の形状で引き上げる

        大規模ネットワークでは密行列が確保できないため、`DENSE_PULL_LIMIT_BYTES` を
        超える場合は COO 版へ誘導する。
        """
        required_bytes = self.total_neurons ** 2 * 4
        if required_bytes > DENSE_PULL_LIMIT_BYTES:
            raise MemoryError(
                f"pull_synapse は {required_bytes / 2**30:.1f} GiB の密行列を確保しようとしました "
                f"(total_neurons={self.total_neurons})。"
                " pull_synapse_coo() / pull_synapse_flat() を使ってください。"
            )

        global_matrix = np.zeros((self.total_neurons, self.total_neurons), dtype=np.float32)

        for syn_pop_name, syn_pop in self.model.synapse_populations.items():
            index = self.builder.synapse_index.get(syn_pop_name)
            if index is None:
                raise KeyError(
                    f"synapse_index に '{syn_pop_name}' がありません。"
                    " NetworkBuilder.build() を経ずに構築されたモデルの可能性があります。"
                )
            syn_pop.vars[var_name].pull_from_device()
            global_matrix[index.global_src, index.global_tgt] = syn_pop.vars[var_name].values

        return global_matrix

    def get_global_spikes(self) -> Dict[str, np.ndarray]:
        """
        全Populationのスパイク記録を収集し、グローバルIDに変換して時間順にソートした結果を返す
        """
        if getattr(self.model, "_recording_in_use", False):
            self.model.pull_recording_buffers_from_device()

        all_times = []
        all_global_ids = []

        for pop_name in self.layout.names():
            # GeNNからローカルデータを直接参照
            times, local_ids = self.model.neuron_populations[pop_name].spike_recording_data[0]

            if len(times) > 0:
                all_times.append(times)
                # ループ内で直接グローバルインデックスにマッピング (連番なので start+local)
                all_global_ids.append(self.layout.local_to_global(pop_name, local_ids))

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
        if self.initial_states is None:
            raise RuntimeError(
                "reset() には初期状態の控えが必要ですが、setup(backup_initial_states=False) で "
                "省略されています。reset() を使うなら setup() を既定 (True) で呼んでください。"
            )
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

        # 4. per-synapse 到着機構 (pre_arrival_syn_code) を使う場合の状態リセット。
        # ソーススパイクキューをクリアし、前トライアル末尾のスパイクが新トライアル最初の
        # maxDelay ステップで偽の到着を生むのを防ぐ + arrST を負センチネルへ戻す。
        # 到着機構を使うシナプス群が無ければ no-op。
        self.model.reset_arrival_state()

        print("  [Simulator] All network variables (Neurons, Synapses, Inputs) safely reset.")

    def _split_global_to_local(self, global_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        [形状: (total_neurons,)] のグローバル配列を Population毎に分割する
        """
        return self.layout.split_global_to_local(global_data)

    def _merge_local_to_global(self, local_dict: Dict[str, np.ndarray], dtype=np.float32) -> np.ndarray:
        """
        Population毎の配列を [形状: (total_neurons,)] のグローバル配列に結合する
        """
        return self.layout.merge_local_to_global(local_dict, dtype=dtype)
    
