import numpy as np
import pygenn
from src.core.registry import SPATIAL_MODELS, CONNECTION_MODELS, WEIGHT_MODELS, DELAY_MODELS, NEURON_MODELS


class NetworkBuilder:
    def __init__(self, config: dict, rng: np.random.RandomState = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.RandomState(config.get("base", {}).get("seed", 42))
        
        self.genn_model = pygenn.GeNNModel("float", "SNN_Model")
        self.genn_model.dt = self.config.get("simulation", {}).get("dt", 0.1)
        
        self.group_info = {}
        # yamlから総ニューロン数を計算
        self.total_neurons = sum(p["num"] for p in self.config.get("neurons", {}).values())

    def build(self) -> pygenn.GeNNModel:
        print("=== ネットワーク構築 (Network Building) ===")
        # 1. グループへのアサイン
        self._assign_neurons()
        # 2. グローバル行列(トポロジー等)の生成
        self._generate_global_matrices()
        # 3. シナプスグループの構築
        self._build_synapses()
        return self.genn_model

    def _assign_neurons(self):
        print(f"  Generating Spatial Coordinates for {self.total_neurons} neurons...")

        # --- 2. 座標(インデックス)のアサイン(割り当て) ---
        available_indices = np.arange(self.total_neurons)
            
        # (※もし「層(Layer)」のように特定座標で分けたい場合は、Z座標でソートしたインデックス等を使用可能)

        # --- 3. グループごとにGeNNへ登録 ---
        current_offset = 0
        for group_name, params in self.config.get("neurons", {}).items():
            num_neurons = params['num']
            
            assigned_indices = self.rng.choice(available_indices, size=num_neurons, replace=False)
            available_indices = np.setdiff1d(available_indices, assigned_indices)
            assigned_indices.sort()
            
            self.group_info[group_name] = {
                "global_indices": assigned_indices,
                "num": num_neurons
            }
            current_offset += num_neurons

            # GeNNへの登録
            NeuronClass = NEURON_MODELS.get(params['type']) 
            init_kwargs = params.copy()
            init_kwargs["dt"] = self.config.get("simulation", {}).get("dt", 0.1)
            neuron_instance = NeuronClass(**init_kwargs)
            ng = self.genn_model.add_neuron_population(
                group_name, 
                num_neurons,
                neuron_instance.model_class, 
                neuron_instance.params, 
                neuron_instance.initial_vars
            )
            print(f"  Added NeuronGroup: {group_name} ({num_neurons} neurons, randomly assigned)")
        print(f"  Added {current_offset} neurons as a result")

    def _generate_global_matrices(self):
        network = self.config.get("network")

        spaceClass = SPATIAL_MODELS.get(network["space"]["type"])
        self.global_coords = spaceClass(network["space"], self.total_neurons, self.rng).generate() 
        
        connectClass = CONNECTION_MODELS.get(network["connection"]["type"])
        self.global_mask = connectClass(network["connection"], self.global_coords, self.rng).generate() 
        
        weightClass = WEIGHT_MODELS.get(network["weight"]["type"])
        self.global_weights = weightClass(network["weight"], self.global_coords, self.global_mask, self.rng).generate()
        
        delayClass = DELAY_MODELS.get(network["delay"]["type"])
        self.global_delays = delayClass(network["delay"], self.global_coords, self.global_mask, self.rng).generate()
        

    def _build_synapses(self):
        for syn_group_name, params in self.config.get("synapse_groups", {}).items():
            src_name = params["source"]
            tgt_name = params["target"]
            
            # グループに割り当てられたグローバルインデックスを取得 (不連続な配列の可能性あり)
            src_indices = self.group_info[src_name]["global_indices"]
            tgt_indices = self.group_info[tgt_name]["global_indices"]

            # --- np.ix_ を使って、グローバル行列から該当する座席同士の結合を抽出 ---
            sub_weights = self.global_weights[np.ix_(src_indices, tgt_indices)].copy()
            sub_mask = self.global_mask[np.ix_(src_indices, tgt_indices)] 

            weight_scale = params.get("weight_scale", 1.0)
            sub_weights *= weight_scale

            # GeNN形式への変換
            local_src_idx, local_tgt_idx = np.where(sub_mask != 0)
            weights_flat = sub_weights[local_src_idx, local_tgt_idx]

            # TODO: 将来的には SYNAPSE_MODELS レジストリを使用
            # ソースとターゲットの NeuronGroup インスタンスを取得
            src_pop = self.genn_model.neuron_populations[src_name]
            tgt_pop = self.genn_model.neuron_populations[tgt_name]

            # GeNN 5の書式に合わせて各種初期化子(init_...)を使用する
            weight_init = pygenn.genn_model.init_weight_update(
                "StaticPulse", 
                {}, 
                {"g": weights_flat}
            )
            post_init = pygenn.genn_model.init_postsynaptic(
                "DeltaCurr", 
                {}
            )

            sg = self.genn_model.add_synapse_population(
                syn_group_name, 
                "SPARSE",        # matrix_type
                src_pop,         # source (NeuronGroup オブジェクト)
                tgt_pop,         # target (NeuronGroup オブジェクト)
                weight_init,     # weight_update_init
                post_init,       # postsynaptic_init
            )
            sg.set_sparse_connections(local_src_idx, local_tgt_idx)
