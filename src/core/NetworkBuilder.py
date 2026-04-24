import numpy as np
import pygenn
from typing import Dict, Any, Tuple
from pathlib import Path
import sys

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Pydanticの設定モデルと、コンポーネントを動的ロードするレジストリをインポート
from src.core.config_manager import AppConfig
from src.core.registry import SPATIAL_MODELS, CONNECTION_MODELS, WEIGHT_MODELS, DELAY_MODELS, NEURON_MODELS, SYNAPSE_MODELS, PLASTICITY_MODELS

class NetworkBuilder:
    def __init__(self, config: AppConfig):
        self.config = config
        self.rng = np.random.RandomState(config.simulation.seed)
        
        self.genn_model = pygenn.GeNNModel("float", "SNN_Model")
        self.genn_model.dt = self.config.simulation.dt
        # self.genn_model.batch_size = self.config.task.batch_size

        # Pydanticモデルから総ニューロン数を計算
        self.total_neurons = sum(neuron_cfg.num for neuron_cfg in self.config.neurons.values())
        
        self.group_info = {}
        self.global_coords = None
        self.global_mask = None
        self.global_weights = None
        self.global_delays = None

    def build(self, rec_spike: bool = True) -> Tuple[pygenn.GeNNModel, Dict[str, Any]]:
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
        
        return self.genn_model, self.group_info

    def _assign_neurons(self):
        """総ニューロン数に対して、グループごとにランダムにグローバルインデックスを割り当てる"""
        print(f"  Assigning Global Indices for {self.total_neurons} neurons...")
        available_indices = np.arange(self.total_neurons)
        current_offset = 0
        self.group_info = {}
        
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
                pop_name = group_name, 
                num_neurons = num_neurons,
                neuron = neuron_instance.model_class, 
                params = neuron_instance.params, 
                vars = neuron_instance.vars
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
            sub_delays = self.global_delays[np.ix_(src_indices, tgt_indices)].copy()
            sub_mask = self.global_mask[np.ix_(src_indices, tgt_indices)] 

            local_src_idx, local_tgt_idx = np.where(sub_mask != 0)
            weights_flat = sub_weights[local_src_idx, local_tgt_idx]
            delays_flat = sub_delays[local_src_idx, local_tgt_idx]
            delays_flat = delays_flat // self.config.simulation.dt

            src_pop = self.genn_model.neuron_populations[src_name]
            tgt_pop = self.genn_model.neuron_populations[tgt_name]

            PlasClass = PLASTICITY_MODELS.get(syn_cfg.plasticity.type)
            plas_instance = PlasClass(
                config=syn_cfg.plasticity, 
                dt=self.config.simulation.dt,
                weight=weights_flat,
                delay=delays_flat
            )

            weight_init = pygenn.genn_model.init_weight_update(
                snippet=plas_instance.model_class, 
                params=plas_instance.params, 
                vars=plas_instance.vars,
                pre_vars=plas_instance.pre_vars,
                post_vars=plas_instance.post_vars,
                pre_var_refs={},
                post_var_refs={},
                psm_var_refs={}
                )
            SynClass = SYNAPSE_MODELS.get(syn_cfg.synapse.type)
            syn_instance = SynClass(syn_cfg.synapse, self.config.simulation.dt)
            post_init = pygenn.genn_model.init_postsynaptic(
                snippet="ExpCurr", 
                params={"tau": 15.0},
                vars={},
                var_refs={}
                )

            sg = self.genn_model.add_synapse_population(
                pop_name=syn_group_name, 
                matrix_type="SPARSE", 
                source=src_pop, 
                target=tgt_pop, 
                weight_update_init=weight_init, 
                postsynaptic_init=post_init
            )
            sg.set_sparse_connections(local_src_idx, local_tgt_idx)

    def _build_input_ports(self):
        """GeNN上に入力専用のglobal_popを定義し、本体へ1対1で接続する"""

        if self.config.inputs.GaussianNoise.enable:
            for pop_name, info in self.group_info.items():
                cs_name = f"GaussianNoise_CS_to_{pop_name}"

                cs = self.genn_model.add_current_source(
                    cs_name=cs_name,
                    current_source_model="GaussianNoise", 
                    pop=self.genn_model.neuron_populations[pop_name],
                    params={
                        "mean": self.config.inputs.GaussianNoise.mean,
                        "sd": self.config.inputs.GaussianNoise.sd
                    },
                    vars={}
                )
                print(f"    Added Gaussian Noise Current Source '{cs_name}' to '{pop_name}'")
            
        print(f"  Added Input Groups for {self.total_neurons} global neurons.")
            

    def _build_output_ports(self, rec_spike: bool):
        """スパイク記録の準備"""
        for group_name in self.config.neurons.keys():
            pop_name = group_name
            pop = self.genn_model.neuron_populations[pop_name]
    
            # スパイクの記録設定
            pop.spike_recording_enabled = rec_spike
                





if __name__ == "__main__":
    import traceback
    from src.core.config_manager import ConfigManager
    import models.neurons.pqn_float
    import models.neurons.pqn_int
    import src.models.network.space
    import src.models.network.connectors
    import src.models.network.weights
    import src.models.network.delays
    import src.data.test_data

    print("=== NetworkBuilder 動作検証テストを開始します ===")

    try:
        # 1. Configのロード
        config_src = "test.yaml" # 実際のファイルパスに合わせてください
        print(f"Loading config from {config_src}...")
        manager = ConfigManager(config_src, "pqn_test") 
        config = manager.resolve()

        # 2. ビルダーの初期化
        print("\n[TEST] Initializing NetworkBuilder...")
        builder = NetworkBuilder(config)
        print(f"  ✓ Initialization successful. Total neurons: {builder.total_neurons}")

        # 3. ビルドプロセスの実行
        print("\n[TEST] Executing build()...")
        genn_model, group_info = builder.build(rec_spike=True)
        
        # 4. アサーションと検証
        print("\n[TEST] Validating generated objects...")
        pop_names = list(genn_model.neuron_populations.keys())
        print(f"  - Registered Populations in GeNN: {pop_names}")
        CSpop_names = list(genn_model.current_sources.keys())
        print(f"  - Registered Current Sources in GeNN: {CSpop_names}")

        # --- 本体層の存在確認 ---
        for body_name in config.neurons.keys():
            assert body_name in pop_names, f"Body Pop '{body_name}' not found in GeNN model."
            assert body_name in group_info.keys(), f"'{body_name}' missing in group_info."
        print("  ✓ Body populations successfully validated.")

        # --- Current Source の確認 ---
        if config.inputs.current_DC.enable:
            cs_names = list(genn_model.current_sources.keys())
            print(f"  - Registered Current Sources: {cs_names}")
            # 各body_popに対してDCソースが作られているか確認
            for body_name in config.neurons.keys():
                expected_cs = f"DC_CS_to_{body_name}"
                assert expected_cs in cs_names, f"Current Source '{expected_cs}' missing."
            print("  ✓ Current sources successfully validated.")

        print("\n🎉 === 全てのテストをクリアしました (All Tests Passed) === 🎉")

    except Exception as e:
        print(f"\n❌ [Error] 検証中にエラーが発生しました:")
        traceback.print_exc()