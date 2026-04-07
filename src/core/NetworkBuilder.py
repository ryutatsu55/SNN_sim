import numpy as np
import pygenn
from typing import Dict, Any, Tuple

# Pydanticの設定モデルと、コンポーネントを動的ロードするレジストリをインポート
from src.core.config_manager import AppConfig
from src.core.registry import SPATIAL_MODELS, CONNECTION_MODELS, WEIGHT_MODELS, DELAY_MODELS, NEURON_MODELS

class NetworkBuilder:
    def __init__(self, config: AppConfig):
        self.config = config
        self.rng = np.random.RandomState(config.simulation.seed)
        
        self.genn_model = pygenn.GeNNModel("float", "SNN_Model")
        self.genn_model.dt = self.config.simulation.dt
        
        # Pydanticモデルから総ニューロン数を計算
        self.total_neurons = sum(neuron_cfg.num for neuron_cfg in self.config.neurons.values())
        
        self.group_info = {}
        self.global_coords = None
        self.global_mask = None
        self.global_weights = None
        self.global_delays = None

        # EncoderとSimulatorに渡すための APIコントラクト（メタデータ）
        self.io_map = {
            "inputs": {},
            "outputs": {},
            "meta": {}
        }

    def build(self, rec_spike: bool = False) -> Tuple[pygenn.GeNNModel, Dict[str, Any]]:
        print("=== ネットワーク構築 (Network Building) ===")
        # 1. 空間座席（インデックス）のランダム割り当て
        self._assign_neurons()
        # 2. 全ニューロン一括での座標・グローバル行列生成
        self._generate_global_matrices()
        # 3. GeNNへのニューロン・シナプスの登録
        self._build_neuron_populations()
        self._build_synapses()
        # 4. 入出力ポートのハードウェア的構築とメタデータ生成
        self._build_input_ports()
        self._build_output_ports(rec_spike)

        self.io_map["meta"] = {
            "total_neurons": self.total_neurons,
            "global_coords": self.global_coords 
        }
        
        return self.genn_model, self.io_map

    def _assign_neurons(self):
        """総ニューロン数に対して、グループごとにランダムにグローバルインデックスを割り当てる"""
        print(f"  Assigning Global Indices for {self.total_neurons} neurons...")
        available_indices = np.arange(self.total_neurons)
        current_offset = 0
        
        for group_name, params in self.config.neurons.items():
            num_neurons = params.num
            
            # ランダムに座席を確保し、ソートする
            assigned_indices = self.rng.choice(available_indices, size=num_neurons, replace=False)
            available_indices = np.setdiff1d(available_indices, assigned_indices)
            assigned_indices.sort()
            
            self.group_info[group_name] = {
                "global_indices": assigned_indices,
                "num": num_neurons,
                "params": params
            }
            current_offset += num_neurons
            
        print(f"  Successfully assigned {current_offset} neurons.")

    def _generate_global_matrices(self):
        """全ニューロンの座標と、グローバルな結合行列を一括生成する"""
        print("  Generating Global Coordinates and Matrices...")
        network = self.config.network

        # Pydanticモデルから辞書を取得
        space_cfg = network.space
        conn_cfg = network.connection
        weight_cfg = network.weight
        delay_cfg = network.delay
        # ======================================================================================
        # ニューロン数が数万単位の場合、scipy.sparse.csr_matrixを使用して疎行列として生成することも検討
        # ======================================================================================
        # 1. 空間座標の生成
        spaceClass = SPATIAL_MODELS.get(space_cfg.profile_name)
        self.global_coords = spaceClass(space_cfg, self.total_neurons, self.rng).generate() 
        
        # 2. 結合マスクの生成
        connectClass = CONNECTION_MODELS.get(conn_cfg.profile_name)
        self.global_mask = connectClass(conn_cfg, self.total_neurons, self.global_coords, self.rng).generate() 
        
        # 3. 重み行列の生成
        weightClass = WEIGHT_MODELS.get(weight_cfg.profile_name)
        self.global_weights = weightClass(weight_cfg, self.total_neurons, self.global_coords, self.global_mask, self.rng).generate()
        
        # 4. 遅延行列の生成
        delayClass = DELAY_MODELS.get(delay_cfg.profile_name)
        self.global_delays = delayClass(delay_cfg, self.total_neurons, self.global_coords, self.global_mask, self.rng).generate()

    def _build_neuron_populations(self):
        """GeNN上にニューロンポピュレーションを定義"""
        for group_name, info in self.group_info.items():
            params = info["params"]
            num_neurons = info["num"]
            
            NeuronClass = NEURON_MODELS.get(params.type) 
            neuron_instance = NeuronClass(params, self.config.simulation.dt)
            
            self.genn_model.add_neuron_population(
                group_name, 
                num_neurons,
                neuron_instance.model_class, 
                neuron_instance.params, 
                neuron_instance.initial_vars
            )
            print(f"  Added NeuronGroup: {group_name} ({num_neurons} neurons)")

    def _build_synapses(self):
        """グローバル行列からインデックスで切り出し、シナプスを構築"""
        print("  Building Synapse Populations...")
        for syn_group_name, syn_cfg in self.config.synapses.items():
            src_name = syn_cfg.source
            tgt_name = syn_cfg.target
            
            src_indices = self.group_info[src_name]["global_indices"]
            tgt_indices = self.group_info[tgt_name]["global_indices"]

            # グローバル行列からの抽出
            sub_weights = self.global_weights[np.ix_(src_indices, tgt_indices)].copy()
            sub_mask = self.global_mask[np.ix_(src_indices, tgt_indices)] 

            weight_scale = syn_cfg.weight_scale
            sub_weights *= weight_scale

            local_src_idx, local_tgt_idx = np.where(sub_mask != 0)
            weights_flat = sub_weights[local_src_idx, local_tgt_idx]

            src_pop = self.genn_model.neuron_populations[src_name]
            tgt_pop = self.genn_model.neuron_populations[tgt_name]

            weight_init = pygenn.genn_model.init_weight_update("StaticPulse", {}, {"g": weights_flat})
            post_init = pygenn.genn_model.init_postsynaptic("DeltaCurr", {})

            sg = self.genn_model.add_synapse_population(
                syn_group_name, "SPARSE", src_pop, tgt_pop, weight_init, post_init
            )
            sg.set_sparse_connections(local_src_idx, local_tgt_idx)

    def _build_input_ports(self):
        """データ入出力用のポートとメタデータを構築"""
        for group_name, detail in self.config.neurons.items():
            
            global_indices = self.group_info[group_name]["global_indices"]
            coords = self.global_coords[global_indices]
            target_var = detail.in_var

            self.io_map["inputs"][group_name] = {
                "target_pop": group_name,
                "target_var": target_var,
                "global_indices": global_indices,
                "coords": coords
            }
            

    def _build_output_ports(self, rec_spike: bool):
        """記録変数のメタデータを構築"""
        for group_name, detail in self.config.neurons.items():
            pop_name = group_name
            record_var = detail.out_var

            self.io_map["outputs"][pop_name] = {
                "record_vars": [record_var]
            }
            pop = self.genn_model.neuron_populations[pop_name]
    
            # 1. スパイクの記録設定
            if rec_spike:
                self.io_map["outputs"][pop_name]["record_vars"].append("spikes")
                pop.spike_recording_enabled = True
                
            # 2. 膜電位(V)などの連続変数の記録設定
            pop.vars[record_var].record = True
