import os
import inspect
import numpy as np
import pygenn
from typing import Dict, Any, Tuple
from pathlib import Path
import sys

# CUDAバックエンドを有効にするため、未設定の場合はデフォルトパスを補完する
if not os.environ.get("CUDA_PATH"):
    for _candidate in ["/usr/local/cuda", "/usr/cuda"]:
        if Path(_candidate).exists():
            os.environ["CUDA_PATH"] = _candidate
            break

# CPU/GPU切り替え: False にすると single_threaded_cpu バックエンドを使用
USE_GPU = False

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Pydanticの設定モデルと、コンポーネントを動的ロードするレジストリをインポート
from src.core.config_manager import AppConfig
from src.core.layout import NetworkLayout
from src.core.registry import SPATIAL_MODELS, CONNECTION_MODELS, WEIGHT_MODELS, DELAY_MODELS, NEURON_MODELS, SYNAPSE_MODELS, PLASTICITY_MODELS

class NetworkBuilder:
    def __init__(self, config: AppConfig, model_name: str = "SNN_Model", code_gen_dir: str | None = None):
        self.config = config
        self.rng = np.random.RandomState(config.simulation.seed)

        # GeNN が生成する <model_name>_CODE を置く親ディレクトリ。
        # None の場合はカレントディレクトリ (従来挙動)。Simulator.setup() の build() で使う。
        self.code_gen_dir = code_gen_dir

        _backend = "cuda" if USE_GPU else "single_threaded_cpu"
        self.genn_model = pygenn.GeNNModel("double", model_name, time_precision="double", backend=_backend)
        self.genn_model.dt = self.config.simulation.dt
        # self.genn_model.batch_size = self.config.task.batch_size

        # config のニューロン宣言順に連番でグローバルインデックスを割り当てる決定論的レイアウト。
        # RandomState を消費しないため、GeNN ビルドなしでも from_config だけで再現できる。
        self.layout = NetworkLayout.from_config(config)
        self.total_neurons = self.layout.total_neurons

        self._component_lifeline = []
        self.global_coords = None
        self.global_mask = None
        self.global_weights = None
        self.global_delays = None

    def build(self, rec_spike: bool = True) -> Tuple[pygenn.GeNNModel, NetworkLayout]:
        print("=== ネットワーク構築 (Network Building) ===")
        # 1. 全ニューロン一括での座標・グローバル行列生成 (レイアウトは __init__ で確定済み)
        self._generate_global_matrices()
        # 2. GeNNへのニューロン・シナプスの登録
        self._build_neuron_populations()
        self._build_synapses()
        # 3. 入出力ポートのハードウェア的構築とメタデータ生成
        self._build_input_ports()
        self._build_output_ports(rec_spike)

        return self.genn_model, self.layout

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
        self.global_coords = spaceClass(space_cfg, self.total_neurons, self.rng, layout=self.layout).generate()

        # 2. 結合マスクの生成
        connectClass = CONNECTION_MODELS.get(conn_cfg.profile_name)
        self.global_mask = connectClass(conn_cfg, self.total_neurons, self.global_coords, self.rng, layout=self.layout).generate()

        # 3. 重み行列の生成
        weightClass = WEIGHT_MODELS.get(weight_cfg.profile_name)
        self.global_weights = weightClass(weight_cfg, self.total_neurons, self.global_coords, self.global_mask, self.rng, layout=self.layout).generate()

        # 4. 遅延行列の生成
        delayClass = DELAY_MODELS.get(delay_cfg.profile_name)
        self.global_delays = delayClass(delay_cfg, self.total_neurons, self.global_coords, self.global_mask, self.rng, layout=self.layout).generate()

    def _build_neuron_populations(self):
        """GeNN上にニューロンポピュレーションを定義"""
        for group_name, grp in self.layout.items():
            params = grp.params
            num_neurons = grp.num

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
            for tgt_name in self.layout.names():
                src_name = syn_cfg.source
                # tgt_name = syn_cfg.target
                print(f"src:{src_name}, tgt:{tgt_name}")

                # 連番割り当てなので集団はグローバル空間の連続スライスに対応する
                src_slice = self.layout.slice_of(src_name)
                tgt_slice = self.layout.slice_of(tgt_name)

                # グローバル行列からの抽出
                sub_weights = self.global_weights[src_slice, tgt_slice].copy()
                sub_delays = self.global_delays[src_slice, tgt_slice].copy()
                sub_mask = self.global_mask[src_slice, tgt_slice]

                delay_by_target = getattr(syn_cfg, "delay_by_target", None)
                # delay_by_target 指定は集団内で単一定数 = 均一遅延。この場合のみ GeNN の
                # 軸索遅延(axonal_delay_steps)を使い、pre_spike_syn_code を到着時刻に
                # イベント駆動化して毎ステップの syn_dynamics_code を撤廃する(高速化)。
                use_axonal = delay_by_target is not None and tgt_name in delay_by_target
                if use_axonal:
                    sub_delays[sub_mask != 0] = float(delay_by_target[tgt_name])
                elif hasattr(syn_cfg, "delay"):
                    sub_delays[sub_mask != 0] = float(syn_cfg.delay)

                local_src_idx, local_tgt_idx = np.where(sub_mask != 0)
                if len(local_src_idx) == 0:
                    print(f"    Skipping SynapseGroup: {src_name}_to_{tgt_name} (No connections found)")
                    continue
                weights_flat = sub_weights[local_src_idx, local_tgt_idx]
                delays_flat = sub_delays[local_src_idx, local_tgt_idx]
                delays_flat = np.rint(delays_flat / self.config.simulation.dt).astype(np.uint8)

                src_pop = self.genn_model.neuron_populations[src_name]
                tgt_pop = self.genn_model.neuron_populations[tgt_name]

                PlasClass = PLASTICITY_MODELS.get(syn_cfg.plasticity.type)
                # 可塑性モデルが axonal_delay_steps を受け取れる場合のみ軸索遅延経路を適用する
                # (= opt-in)。標準モデルは受け取らないため従来の dendritic delay 経路のまま。
                supports_axonal = "axonal_delay_steps" in inspect.signature(PlasClass.__init__).parameters
                axonal_steps = None
                if use_axonal and supports_axonal:
                    uniform_delay_ms = float(delay_by_target[tgt_name])
                    # STDP の実効到着時刻は emission + axonal_delay_steps*dt になる(learn-post /
                    # pre_spike_syn の処理ステップ基準。実測で確認済み)。よって STDP タイミングを
                    # delay_corrected と一致させるには steps = round(D/dt)。
                    # ※電流(addToPost)の到着は (steps+1)*dt となり D より 1dt(=dt)遅いが、
                    #   PSC は指数減衰でならされるため重み学習への影響は無視できる。
                    axonal_steps = max(0, int(round(uniform_delay_ms / self.config.simulation.dt)))
                else:
                    use_axonal = False  # 非対応モデルには適用しない

                plas_kwargs = dict(
                    config=syn_cfg.plasticity,
                    dt=self.config.simulation.dt,
                    weight=weights_flat,
                    delay=delays_flat,
                    num_pre=src_pop.num_neurons,
                    num_post=tgt_pop.num_neurons,
                )
                if use_axonal:
                    plas_kwargs["axonal_delay_steps"] = axonal_steps
                plas_instance = PlasClass(**plas_kwargs)
                self._component_lifeline.append(plas_instance)
                weight_init = pygenn.genn_model.init_weight_update(
                    snippet=plas_instance.snippet, 
                    params=plas_instance.params, 
                    vars=plas_instance.vars,
                    pre_vars=plas_instance.pre_vars,
                    post_vars=plas_instance.post_vars,
                    pre_var_refs=plas_instance.pre_var_refs,
                    post_var_refs=plas_instance.post_var_refs,
                    psm_var_refs=plas_instance.psm_var_refs   
                )
                
                SynClass = SYNAPSE_MODELS.get(syn_cfg.synapse.type)
                syn_instance = SynClass(
                    config=syn_cfg.synapse, 
                    dt=self.config.simulation.dt,
                    pop=tgt_pop
                )
                self._component_lifeline.append(syn_instance)
                post_init = pygenn.genn_model.init_postsynaptic(
                    snippet=syn_instance.snippet, 
                    params=syn_instance.params,
                    vars=syn_instance.vars,
                    var_refs=syn_instance.var_refs
                )

                sg = self.genn_model.add_synapse_population(
                    pop_name=f"{src_name}_to_{tgt_name}", 
                    matrix_type="SPARSE", 
                    source=src_pop, 
                    target=tgt_pop, 
                    weight_update_init=weight_init, 
                    postsynaptic_init=post_init
                )
                sg.set_sparse_connections(local_src_idx, local_tgt_idx)
                if use_axonal:
                    # 軸索遅延: ソーススパイクを axonal_steps 分遅延スロットから読み、到着時刻に
                    # pre_spike_syn_code を発火。addToPost 即時投与なので dendritic バッファは不要。
                    sg.axonal_delay_steps = axonal_steps
                else:
                    max_delay_steps = int(np.max(delays_flat)) + 1
                    sg.max_dendritic_delay_timesteps = max_delay_steps

    def _build_input_ports(self):
        """GeNN上に入力専用のglobal_popを定義し、本体へ1対1で接続する"""

        if self.config.inputs.GaussianNoise.enable:
            for pop_name in self.layout.names():
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
    import src.models.neurons.pqn_float
    import src.models.neurons.pqn_int
    import src.models.neurons.lif
    import src.models.network.space
    import src.models.network.connectors
    import src.models.network.weights
    import src.models.network.delays
    import src.models.plasticity.standard_models
    import src.models.plasticity.custom_Akita
    import src.models.synapses.standard_models
    import src.models.synapses.custom
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
        genn_model, layout = builder.build(rec_spike=True)

        # 4. アサーションと検証
        print("\n[TEST] Validating generated objects...")
        Npop_names = list(genn_model.neuron_populations.keys())
        print(f"  - Registered Populations in GeNN: {Npop_names}")
        Spop_names = list(genn_model.synapse_populations.keys())
        print(f"  - Registered Synapse Populations in GeNN: {Spop_names}")
        CSpop_names = list(genn_model.current_sources.keys())
        print(f"  - Registered Current Sources in GeNN: {CSpop_names}")

        # --- 本体層の存在確認 ---
        for body_name in config.neurons.keys():
            assert body_name in Npop_names, f"Body Pop '{body_name}' not found in GeNN model."
            assert body_name in layout.names(), f"'{body_name}' missing in layout."
        print("  ✓ Body populations successfully validated.")

        # --- Current Source の確認 ---
        if config.inputs.GaussianNoise.enable:
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
